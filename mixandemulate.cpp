#include <chrono>
#include <iostream>
#include <string>
#include <ctime>
#include <sstream>

#include <fstream>
#include <vector>
#include <limits>

#include "Eigen/Dense"
#include <Eigen/StdVector>

#include "crn.h"
#include "tree.h"
#include "brt.h"
#include "brtfuns.h"
#include "dinfo.h"
#include "mbrt.h"
#include "ambrt.h"
#include "psbrt.h"
#include "tnorm.h"
#include "mxbrt.h"
#include "amxbrt.h"

using std::cout;
using std::endl;

#define MODEL_BT 1
#define MODEL_BINOMIAL 2
#define MODEL_POISSON 3
#define MODEL_BART 4
#define MODEL_HBART 5
#define MODEL_PROBIT 6
#define MODEL_MODIFIEDPROBIT 7
#define MODEL_MERCK_TRUNCATED 8
#define MODEL_MIXBART 9


int main(int argc, char* argv[])
{
    std::string folder("");

    if(argc>1)
    {
        std::string confopt("--conf");
    if(confopt.compare(argv[1])==0) {
#ifdef _OPENMPI
        return 101;
#else
        return 100;
#endif
        }

    //otherwise argument on the command line is path to conifg file.
    folder=std::string(argv[1]);
    folder=folder+"/";
    }


    //-----------------------------------------------------------
    //random number generation
    crn gen;
    gen.set_seed(static_cast<long long>(std::chrono::high_resolution_clock::now()
                                    .time_since_epoch()
                                    .count()));

    //--------------------------------------------------
    //process args
    std::ifstream conf(folder+"config");

    // model type
    int modeltype;
    conf >> modeltype;

    //Number of models
    int nummodels;
    conf >> nummodels;

    // extract the cores for the mix model and simulator inputs
    std::string xcore,ycore,score,ccore;
    double means,base,baseh,power,powerh,lam,nu;
    size_t m,mh;

    std::vector<std::string> xcore_list,ycore_list,score_list,ccore_list;
    std::vector<double> means_list,base_list,baseh_list,power_list,powerh_list, lam_list, nu_list;
    std::vector<size_t> m_list, mh_list;
    
    // Get model mixing cores & inputs then get each emulator cores & inputs 
    for(int i=0;i<=nummodels;i++){
        // Get xcore, ycore, score, and ccore
        conf >> xcore;
        conf >> ycore;
        conf >> score;
        conf >> ccore;

        // Store into respective lists
        xcore_list.push_back(xcore);
        ycore_list.push_back(ycore);
        score_list.push_back(score);
        ccore_list.push_back(ccore);

        // Data means
        conf >> means;
        means_list.push_back(means);

        // Tree sizes
        conf >> m;
        conf >> mh;
        m_list.push_back(m);
        mh_list.push_back(mh);

        // Tree prior hyperparameters
        conf >> base;
        conf >> baseh;
        conf >> power;
        conf >> powerh;
        base_list.push_back(base);
        baseh_list.push_back(baseh);
        power_list.push_back(power);
        powerh_list.push_back(powerh);

        // Variance Prior
        conf >> lam;
        conf >> nu;
        lam_list.push_back(lam);
        nu_list.push_back(nu);


        // Prints
        /*
        cout << "xcore = " << xcore << endl;
        cout << "ycore = " << ycore << endl;
        cout << "score = " << score << endl;
        cout << "ccore = " << ccore << endl;
        cout << "data.mean = " << means << endl;
        cout << "m = " << m << endl;
        cout << "mh = " << mh << endl;
        cout << "base = " << base << endl;
        cout << "baseh = " << baseh << endl;
        cout << "power = " << power << endl;
        cout << "powerh = " << powerh << endl;
        cout << "lam = " << lam << endl;
        cout << "nu = " << nu << endl;
        */
    }

    // Get the design columns per emulator
    std::vector<std::vector<size_t>> x_cols_list(nummodels, std::vector<size_t>(1));
    std::vector<size_t> xcols, pvec;
    size_t p, xcol;
    for(int i=0;i<nummodels;i++){
        conf >> p;
        pvec.push_back(p);
        x_cols_list[i].resize(p);
        for(size_t j = 0; j<p; j++){
            conf >> xcol;
            x_cols_list[i][j] = xcol;
        }

    }

    // Get the id root computer and field obs ids per emulator
    // std::string idcore;
    // conf >> idcore;

    // MCMC properties
    size_t nd, nburn, nadapt, adaptevery;
    conf >> nd;
    conf >> nburn;
    conf >> nadapt;
    conf >> adaptevery;

    // Get tau and beta for terminal node priors
    std::vector<double> tau_emu_list;
    double tau_disc, tau_wts, tau_emu, beta_disc, beta_wts;
    conf >> tau_disc; //discrepancy tau
    conf >> tau_wts; // wts tau
    for(int i=0;i<nummodels;i++){
        conf >> tau_emu; // emulator tau
        tau_emu_list.push_back(tau_emu);
    }
    conf >> beta_disc;
    conf >> beta_wts;

    //control
    double pbd, pb, pbdh, pbh;
    double stepwpert, stepwperth;
    double probchv, probchvh;
    int tc;
    size_t minnumbot;
    size_t minnumboth;
    size_t printevery;
    std::string xicore;
    std::string modelname;
    conf >> pbd;
    conf >> pb;
    conf >> pbdh;
    conf >> pbh;
    conf >> stepwpert;
    conf >> stepwperth;
    conf >> probchv;
    conf >> probchvh;
    conf >> minnumbot;
    conf >> minnumboth;
    conf >> printevery;
    conf >> xicore;
    conf >> tc;
    conf >> modelname;

    bool dopert=true;
    bool doperth=true;
    if(probchv<0) dopert=false;
    if(probchvh<0) doperth=false;
    
    // summary statistics yes/no
    bool summarystats = false;
    std::string summarystats_str;
    conf >> summarystats_str;
    if(summarystats_str=="TRUE"){ summarystats = true; }
    conf.close();

    //MPI initialization
    int mpirank=0;

#ifdef _OPENMPI
    int mpitc;
    MPI_Init(NULL,NULL);
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Comm_rank(MPI_COMM_WORLD,&mpirank);
    MPI_Comm_size(MPI_COMM_WORLD,&mpitc);
#ifndef SILENT
    cout << "\nMPI: node " << mpirank << " of " << mpitc << " processes." << endl;
#endif
    if(tc<=1) return 0; //need at least 2 processes!
    if(tc!=mpitc) return 0; //mismatch between how MPI was started and how the data is prepared according to tc.
    // #else
    //    if(tc!=1) return 0; //serial mode should have no slave threads!
#endif

    //--------------------------------------------------
    // Banner
    if(mpirank==0) {
        cout << endl;
        cout << "-----------------------------------" << endl;
        cout << "OpenBT model mixing interface" << endl;
        cout << "Loading config file at " << folder << endl;
    }

    //--------------------------------------------------
    //read in y for mixing and z's for emulation
    std::vector<std::vector<double>> y_list(ycore_list.size(), std::vector<double>(1));
    std::vector<double> y;
    std::vector<size_t> nvec;
    double ytemp;
    size_t n=0;
    std::stringstream yfss;
    std::string yfs;
    std::ifstream yf;

    for(size_t i=0;i<ycore_list.size();i++){
    if(y.size()>0){y.clear();} //clear the contents of the y vector
    #ifdef _OPENMPI
        if(mpirank>0) { //only load data on slaves
    #endif
        yfss << folder << ycore_list[i] << mpirank;
        yfs=yfss.str();
        yf.open(yfs);
        while(yf >> ytemp)
            y.push_back(ytemp);
        n=y.size();
        // Store into the vectors
        nvec.push_back(n);
        y_list[i].resize(n);
        y_list[i] = y;
        // Append the field obs to the computer ouput when i > 0....not this second portion of the vector will be updated each iteration of mcmc
        if(i>0){
            y_list[i].insert(y_list[i].end(),y_list[0].begin(),y_list[0].end());    
        }

        //reset stream variables
        yfss.str("");
        yf.close();
    
    #ifndef SILENT
        cout << "node " << mpirank << " loaded " << n << " from " << yfs <<endl;
    #endif
    #ifdef _OPENMPI
    }
    #endif
    }

    //--------------------------------------------------
    //read in x 
    std::vector<std::vector<double>> x_list(xcore_list.size(), std::vector<double>(1));
    std::vector<double> x;
    std::stringstream xfss;
    std::string xfs;
    std::ifstream xf;     
    double xtemp;
    p = 0;
    for(size_t i = 0;i<xcore_list.size();i++){
#ifdef _OPENMPI
        if(mpirank>0) {
#endif
        if(x.size() > 0){x.clear();}
        xfss << folder << xcore_list[i] << mpirank;
        xfs=xfss.str();
        xf.open(xfs);
        while(xf >> xtemp){
            x.push_back(xtemp);
            //cout << "model = " << i << "---- x = " << xtemp << endl;
        }
        p = x.size()/nvec[i];
        if(i == 0){
            pvec.insert(pvec.begin(), p);
        }
        // Store into the vectors
        x_list[i].resize(nvec[i]*pvec[i]);
        x_list[i] = x;

        //Update nvec[i] when i>0 (this accounts for the step of adding field obs to the emulation data sets, which happens later)
        if(i>0){
            nvec[i] = nvec[i] + nvec[0];
        }

        //reset stream variables
        xfss.str("");
        xf.close();
#ifndef SILENT
        cout << "node " << mpirank << " loaded " << n << " inputs of dimension " << p << " from " << xfs << endl;
#endif
#ifdef _OPENMPI
        }
        int tempp = (unsigned int) pvec[i];
        MPI_Allreduce(MPI_IN_PLACE,&tempp,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD);
        if(mpirank>0 && p != ((size_t) tempp)) { cout << "PROBLEM LOADING DATA" << endl; MPI_Finalize(); return 0;}
        p=(size_t)tempp;
#endif
        }

    //--------------------------------------------------
    //dinfo
    std::vector<dinfo> dinfo_list(nummodels+1);
    for(int i=0;i<=nummodels;i++){
        dinfo_list[i].n=0;dinfo_list[i].p=pvec[i],dinfo_list[i].x = NULL;dinfo_list[i].y=NULL;dinfo_list[i].tc=tc;
#ifdef _OPENMPI
        if(mpirank>0){ 
#endif 
            dinfo_list[i].n=nvec[i]; dinfo_list[i].x = &x_list[i][0]; dinfo_list[i].y = &y_list[i][0];
#ifdef _OPENMPI
        }
#endif
    }

    //--------------------------------------------------
    //read in sigmav  -- same as above.
    std::vector<std::vector<double>> sigmav_list(score_list.size(), std::vector<double>(1));
    std::vector<double> sigmav;
    std::vector<size_t> nsigvec;
    std::vector<dinfo> disig_list(nummodels+1);
    std::vector<double*> sig_vec(nummodels+1);
    std::stringstream sfss;
    std::string sfs;
    std::ifstream sf;
    double stemp;
    size_t nsig=0;
    for(int i=0;i<=nummodels;i++){
#ifdef _OPENMPI
        if(mpirank>0) { //only load data on slaves
#endif
        sigmav.clear(); // clear the vector of any contents
        sfss << folder << score_list[i] << mpirank;
        sfs=sfss.str();
        sf.open(sfs);
        while(sf >> stemp)
            sigmav.push_back(stemp);
        nsig=sigmav.size();
        // Store the results in the vector
        sigmav_list[i].resize(nsig);
        sigmav_list[i] = sigmav;
        //reset stream variables
        sfss.str("");
        sf.close();
#ifndef SILENT
        cout << "node " << mpirank << " loaded " << nsig << " from " << sfs <<endl;
#endif
#ifdef _OPENMPI
        if(nvec[i]!=nsig) { cout << "PROBLEM LOADING SIGMAV" << endl; MPI_Finalize(); return 0; }
        }
#else
        if(nvec[i]!=nsig) { cout << "PROBLEM LOADING SIGMAV" << endl; return 0; }
#endif

        sig_vec[i]=&sigmav_list[i][0];
        disig_list[i].n=0; disig_list[i].p=pvec[i]; disig_list[i].x=NULL; disig_list[i].y=NULL; disig_list[i].tc=tc;
#ifdef _OPENMPI
        if(mpirank>0) { 
#endif
            disig_list[i].n=nvec[i]; disig_list[i].x=&x_list[i][0]; disig_list[i].y=sig_vec[i];
#ifdef _OPENMPI
        }
#endif
    }
    
    //--------------------------------------------------
    // read in the initial change of variable rank correlation matrix
    std::vector<std::vector<std::vector<double>>> chgv_list;
    std::vector<std::vector<double>> chgv;
    std::vector<double> cvvtemp;
    double cvtemp;
    std::stringstream chgvfss;
    std::string chgvfs;
    std::ifstream chgvf;
    std::vector<int*> lwr_vec(nummodels+1, new int[tc]);
    std::vector<int*> upr_vec(nummodels+1, new int[tc]);

    for(int k=0;k<=nummodels;k++){
        chgvfss << folder << ccore_list[k] << mpirank;
        chgvfs=chgvfss.str();
        chgvf.open(chgvfs);
        for(size_t i=0;i<dinfo_list[k].p;i++) {
            cvvtemp.clear();
            for(size_t j=0;j<dinfo_list[k].p;j++) {
                chgvf >> cvtemp;
                cvvtemp.push_back(cvtemp);
            }
            chgv.push_back(cvvtemp);
        }
        chgv_list.push_back(chgv);
        //reset stream variables
        chgvfss.str("");
        chgvf.close();
#ifndef SILENT
    cout << "mpirank=" << mpirank << ": change of variable rank correlation matrix loaded:" << endl;
#endif
    }
    // if(mpirank==0) //print it out:
    //    for(size_t i=0;i<di.p;i++) {
    //       for(size_t j=0;j<di.p;j++)
    //          cout << "(" << i << "," << j << ")" << chgv[i][j] << "  ";
    //       cout << endl;
    //    }

    //--------------------------------------------------
    // decide what variables each slave node will update in change-of-variable proposals.
#ifdef _OPENMPI
    //int* lwr=new int[tc];
    //int* upr=new int[tc];
    for(int j=0;j<=nummodels;j++){
        lwr_vec[j][0]=-1; upr_vec[j][0]=-1;
        for(size_t i=1;i<(size_t)tc;i++) { 
            lwr_vec[j][i]=-1; upr_vec[j][i]=-1; 
            calcbegend(p,i-1,tc-1,&lwr_vec[j][i],&upr_vec[j][i]);
            if(p>1 && lwr_vec[j][i]==0 && upr_vec[j][i]==0) { lwr_vec[j][i]=-1; upr_vec[j][i]=-1; }
        }
#ifndef SILENT
        if(mpirank>0) cout << "Slave node " << mpirank << " will update variables " << lwr_vec[j][mpirank] << " to " << upr_vec[j][mpirank]-1 << endl;
#endif
    }
#endif
    
    //--------------------------------------------------
    //make xinfo
    std::vector<xinfo> xi_list(nummodels+1);
    std::vector<double> xivec;
    std::stringstream xifss;
    std::string xifs;
    std::ifstream xif;

    for(int j=0;j<=nummodels;j++){
        xi_list[j].resize(pvec[j]);
        for(size_t i=0;i<pvec[j];i++) {
            double xitemp;
            xifss << folder << xicore << (i+1);
            xifs=xifss.str();
            xif.open(xifs);
            while(xif >> xitemp)
                xivec.push_back(xitemp);
            xi_list[j][i]=xivec;
        }
#ifndef SILENT
        cout << "&&& made xinfo\n";
#endif

    //summarize input variables:
#ifndef SILENT
        for(size_t i=0;i<p;i++){
            cout << "Variable " << i << " has numcuts=" << xi_list[j][i].size() << " : ";
            cout << xi_list[j][i][0] << " ... " << xi_list[j][i][xi_list[j][i].size()-1] << endl;
        }
#endif
    }
    
    //--------------------------------------------------
    //Load master list of ids for emulators, designates field obs vs computer output
    /*
    std::vector<std::vector<std::string>> id_list(nummodels);
    std::stringstream idfss;
    std::string idfs;
    std::ifstream idf;
    std::string idtemp;
    
    idfss << folder << idcore << mpirank;
    idfs=idfss.str();
    idf.open(idfs);
    for(int i=0;i<nummodels;i++){
        for(int j=0;j<nvec[i+1];j++){
            idf >> idtemp;
            id_list[i].push_back(idtemp);
        }
    }
    

    //--------------------------------------------------
    //Copy the original z value for all field obs -- do per emulation dataset
    //This is essential when getting the contribution of the field data for the emulator
    //By design, the original z's for field obs are the observed y values.
    std::vector<std::vector<double>> zf_list(nummodels); //Copy of the original observed data
    std::vector<std::vector<int>> zfidx_list(nummodels); //contains row indexes corresponding to whcih row each field obs belongs in per emu dataset
    std::vector<std::vector<double>> xf_emu_list(xcore_list.size(), std::vector<double>(1)); //copy of the x's for the field obs
    for(int i=0;i<nummodels;i++){
        for(int j=0;j<id_list[i].size();j++){
            if(id_list[i][j] == "f"){
                zf_list[i].push_back(y_list[i+1][j]);
                zfidx_list[i].push_back(j);
                for(int k=0;k<pvec[i+1];k++){
                    xf_emu_list[i].push_back(x_list[i][j*pvec[i+1] + k]);
                }
            }
        }
    }
    */
    //--------------------------------------------------
    //Set up model objects and MCMC
    //--------------------------------------------------   
    std::vector<ambrt> ambm_list; // additive mean bart emulators
    std::vector<psbrt> psbm_list; // product variance models for emulators 
    amxbrt axb(m_list[0]); // additive mean mixing bart
    psbrt pxb(mh_list[0],lam_list[0]); //product model for mixing variance
    std::vector<dinfo> dips_list(nummodels+1); //dinfo for psbrt objects
    std::vector<double*> r_list(nummodels+1); //residual list
    double opm; //variance info
    double lambda; //variance info
    finfo fi;
    int l = 0; //Used for indexing emulators
    nu = 1.0; //reset nu to 1, previosuly defined earlier in program

    //Initialize the model mixing bart objects
    if(mpirank > 0){
        fi = mxd::Ones(nvec[0], nummodels+1); //dummy initialize to matrix of 1's -- n0 x K+1 (1st column is discrepancy)
    }
    //cutpoints
    axb.setxi(&xi_list[0]);   
    //function output information
    axb.setfi(&fi, nummodels+1);
    //data objects
    axb.setdata_mix(&dinfo_list[0]);  //set the data
    //thread count
    axb.settc(tc-1);      //set the number of slaves when using MPI.
    //mpi rank
#ifdef _OPENMPI
    axb.setmpirank(mpirank);  //set the rank when using MPI.
    axb.setmpicvrange(lwr_vec[0],upr_vec[0]); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
    //tree prior
    axb.settp(base_list[0], //the alpha parameter in the tree depth penalty prior
            power_list[0]     //the beta parameter in the tree depth penalty prior
            );
    //MCMC info
    axb.setmi(
            pbd,  //probability of birth/death
            pb,  //probability of birth
            minnumbot,    //minimum number of observations in a bottom node
            dopert, //do perturb/change variable proposal?
            stepwpert,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            probchv,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv_list[0]  //initialize the change of variable correlation matrix.
            );

    //Set prior information
    mxd prior_precision(nummodels+1,nummodels+1);
    vxd prior_mean(nummodels+1);
    prior_precision = (1/(tau_wts*tau_wts))*mxd::Identity(nummodels+1,nummodels+1);
    prior_precision(0,0) = (1/(tau_disc*tau_disc));
    prior_mean = beta_wts*vxd::Ones(nummodels+1);
    prior_mean(0) = beta_disc;
    
    //Sets the model priors for the functions if they are different
    axb.setci(prior_precision, prior_mean, sig_vec[0]);

    //--------------------------------------------------
    //setup psbrt object
    //make di for psbrt object
    dips_list[0].n=0; dips_list[0].p=pvec[0]; dips_list[0].x=NULL; dips_list[0].y=NULL; dips_list[0].tc=tc;
#ifdef _OPENMPI
    if(mpirank>0) {
#endif
    r_list[0] = new double[n];
    for(size_t i=0;i<nvec[0];i++) r_list[0][i]=sigmav_list[0][i];
    dips_list[0].x=&x_list[0][0]; dips_list[0].y=r_list[0]; dips_list[0].n=nvec[0];
#ifdef _OPENMPI
    }
#endif

    //Variance infomration
    opm=1.0/((double)mh_list[0]);
    nu=2.0*pow(nu_list[0],opm)/(pow(nu_list[0],opm)-pow(nu_list[0]-2.0,opm));
    lambda=pow(lam_list[0],opm);
 
    //cutpoints
    pxb.setxi(&xi_list[0]);    //set the cutpoints for this model object
    //data objects
    pxb.setdata(&dips_list[0]);  //set the data
    //thread count
    pxb.settc(tc-1); 
    //mpi rank
#ifdef _OPENMPI
    pxb.setmpirank(mpirank);  //set the rank when using MPI.
    pxb.setmpicvrange(lwr_vec[0],upr_vec[0]); //range of variables each slave node will update in MPI change-of-var proposals.
#endif
    //tree prior
    pxb.settp(baseh_list[0], //the alpha parameter in the tree depth penalty prior
            powerh_list[0]     //the beta parameter in the tree depth penalty prior
            );
    pxb.setmi(
            pbdh,  //probability of birth/death
            pbh,  //probability of birth
            minnumboth,    //minimum number of observations in a bottom node
            doperth, //do perturb/change variable proposal?
            stepwperth,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
            probchvh,  //probability of doing a change of variable proposal.  perturb prob=1-this.
            &chgv_list[0]  //initialize the change of variable correlation matrix.
            );
    pxb.setci(nu,lambda);
    
    //Initialize the emulation bart objects
    for(int j=1;j<=nummodels;j++){
        //Iniitialize the jth emulator
        ambm_list.emplace_back(m_list[j]);
        //cutpoints
        ambm_list[l].setxi(&xi_list[j]);    //set the cutpoints for this model object        
        //data objects
        ambm_list[l].setdata(&dinfo_list[j]);
        //thread count
        ambm_list[l].settc(tc-1);      //set the number of slaves when using MPI.
        //mpi rank
    #ifdef _OPENMPI
        ambm_list[l].setmpirank(mpirank);  //set the rank when using MPI.
        ambm_list[l].setmpicvrange(lwr_vec[j],upr_vec[j]); //range of variables each slave node will update in MPI change-of-var proposals.
    #endif
        //tree prior
        ambm_list[l].settp(base_list[j], //the alpha parameter in the tree depth penalty prior
                           power_list[j]     //the beta parameter in the tree depth penalty prior
                        );
        
        //MCMC info
        ambm_list[l].setmi(
                pbd,  //probability of birth/death
                pb,  //probability of birth
                minnumbot,    //minimum number of observations in a bottom node
                dopert, //do perturb/change variable proposal?
                stepwpert,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
                probchv,  //probability of doing a change of variable proposal.  perturb prob=1-this.
                &chgv_list[j]  //initialize the change of variable correlation matrix.
                );
        ambm_list[l].setci(tau_emu_list[l],sig_vec[j]);
    
        //--------------------------------------------------
        //setup psbrt object
        psbm_list.emplace_back(mh_list[j],lam_list[j]);

        //make di for psbrt object
        opm=1.0/((double)mh_list[j]);
        nu=2.0*pow(nu_list[j],opm)/(pow(nu_list[j],opm)-pow(nu_list[j]-2.0,opm));
        lambda=pow(lam_list[j],opm);

        //make dips info
        dips_list[j].n=0; dips_list[j].p=p; dips_list[j].x=NULL; dips_list[j].y=NULL; dips_list[j].tc=tc;
#ifdef _OPENMPI
        if(mpirank>0) {
#endif
        r_list[j] = new double[nvec[j]];
        for(size_t i=0;i<nvec[j];i++) r_list[j][i]=sigmav_list[j][i];
        dips_list[j].x=&x_list[j][0]; dips_list[j].y=r_list[j]; dips_list[j].n=nvec[j];
#ifdef _OPENMPI
        }
#endif
        //cutpoints
        psbm_list[l].setxi(&xi_list[j]);    //set the cutpoints for this model object
        //data objects
        psbm_list[l].setdata(&dips_list[j]);  //set the data
        //thread count
        psbm_list[l].settc(tc-1); 
        //mpi rank
        #ifdef _OPENMPI
        psbm_list[l].setmpirank(mpirank);  //set the rank when using MPI.
        psbm_list[l].setmpicvrange(lwr_vec[j],upr_vec[j]); //range of variables each slave node will update in MPI change-of-var proposals.
        #endif
        //tree prior
        psbm_list[l].settp(baseh_list[j], //the alpha parameter in the tree depth penalty prior
                powerh_list[j]     //the beta parameter in the tree depth penalty prior
                );
        psbm_list[l].setmi(
                pbdh,  //probability of birth/death
                pbh,  //probability of birth
                minnumboth,    //minimum number of observations in a bottom node
                doperth, //do perturb/change variable proposal?
                stepwperth,  //initialize stepwidth for perturb proposal.  If no adaptation it is always this.
                probchvh,  //probability of doing a change of variable proposal.  perturb prob=1-this.
                &chgv_list[j]  //initialize the change of variable correlation matrix.
                );
        psbm_list[l].setci(nu,lambda);
    }

    //-------------------------------------------------- 
    // MCMC
    //-------------------------------------------------- 
    // Define containers -- similar to those in cli.cpp, except now we iterate over K+1 bart objects
    std::vector<int> onn(nd*m*(nummodels+1),1);
    std::vector<std::vector<int> > oid(nd*m*(nummodels+1), std::vector<int>(1));
    std::vector<std::vector<int> > ovar(nd*m*(nummodels+1), std::vector<int>(1));
    std::vector<std::vector<int> > oc(nd*m*(nummodels+1), std::vector<int>(1));
    std::vector<std::vector<double> > otheta(nd*m*(nummodels+1), std::vector<double>(1));
    std::vector<int> snn(nd*mh*(nummodels+1),1);
    std::vector<std::vector<int> > sid(nd*mh*(nummodels+1), std::vector<int>(1));
    std::vector<std::vector<int> > svar(nd*mh*(nummodels+1), std::vector<int>(1));
    std::vector<std::vector<int> > sc(nd*mh*(nummodels+1), std::vector<int>(1));
    std::vector<std::vector<double> > stheta(nd*mh*(nummodels+1), std::vector<double>(1));
    
    // Method Wrappers
    brtMethodWrapper faxb(&brt::f,axb);
    brtMethodWrapper fpxb(&brt::f,pxb);
    std::vector<brtMethodWrapper> fambm;
    std::vector<brtMethodWrapper> fpsbm; 

    for(int i=0;i<nummodels;i++){
        fambm.emplace_back(&brt::f,ambm_list[i]);
        fpsbm.emplace_back(&brt::f,psbm_list[i]);
    }

    // dinfo for predictions
    std::vector<dinfo> dimix_list(nummodels);
    std::vector<double*> fmix_list(nummodels);
    std::vector<std::vector<double>> xf_list(nummodels);
    size_t xcolsize = 0;
    xcol = 0;
    for(int i=0;i<nummodels;i++){
        // Initialize class objects
        fmix_list[i] = new double[nvec[0]];
        dimix_list[i].y=fmix_list[i];
        dimix_list[i].p = pvec[i+1]; 
        dimix_list[i].n=nvec[0]; 
        dimix_list[i].tc=1;
        // Get the appropriate x columns
        xcolsize = x_cols_list[i].size();
        //xf_list[i].resize(nvec[0]*xcolsize);
        for(size_t j=0;j<nvec[0];j++){
            for(size_t k=0;k<xcolsize;k++){
                xcol = x_cols_list[i][k];
                xf_list[i].push_back(x_list[0][j*xcolsize + xcol]);
                //cout << "model = " << i << "x = " << x_list[i][j*xcolsize + xcol] << endl;
            }
        }
        // Set the x component
        if(mpirank > 0){
            dimix_list[i].x = &xf_list[i][0];
        }else{
            dimix_list[i].x = NULL;
        }
        
        // Add the field obs data to the computer obs data (x_list is not used in dimix but it is convenient to update here in this loop)
        x_list[i].insert(x_list[i].end(),xf_list[0].begin(),xf_list[0].end());    
        
    }

    // Items used to extract weights from the mix bart model
    mxd wts_iter(nummodels+1,nvec[0]); //matrix to store current set of weights
    std::vector<double> mixprednotj(nvec[0],0);
    
    double *fw = new double[nvec[0]];
    dinfo diw;
    diw.x = &x_list[0][0]; diw.y=fw; diw.p = pvec[0]; diw.n=nvec[0]; diw.tc=1;

    /*
    // Initialize objects used to extract wts from mix bart and get weighted field obs for emulation
    // Also dinfo for getting predictions from each emulator
    std::vector<double*> fw_list(nummodels);
    std::vector<dinfo> diw_list(nummodels);
    //std::vector<double*> femu_list(nummodels);
    //std::vector<dinfo> diemu_list(nummodels);
    for(int i=0;i<nummodels;i++){ 
        fw_list[i] = new double[nvec[0]];
        diw_list[i].x = &x_list[0][0]; diw_list[i].y=fw_list[i]; diw_list[i].p = pvec[i+1]; diw_list[i].n=y_list[i].size(); diw_list[i].tc=1;
        //femu_list[i] = new double[zf_list[i].size()];
        //diemu_list[i].x = &xf_emu_list[i][0]; diw_list[i].y=femu_list[i]; diw_list[i].p = pvec[i+1]; diw_list[i].n=zf_list[i].size(); diw_list[i].tc=1;
    }
    */
    // Start the MCMC
#ifdef _OPENMPI
    double tstart=0.0,tend=0.0;
    if(mpirank==0) tstart=MPI_Wtime();
    if(mpirank==0) cout << "Starting MCMC..." << endl;
#else
    cout << "Starting MCMC..." << endl;
#endif

    // Initialize finfo using predictions from each emulator
    for(int j=0;j<nummodels;j++){
        ambm_list[j].predict(&dimix_list[j]);
        for(int k=0;k<nvec[0];k++){
            fi(k,j+1) = fmix_list[j][k] + means_list[j+1]; 
        }   
    }

    // Adapt Stage in the MCMC
    for(size_t i=0;i<nadapt;i++) { 
        // Print adapt step number
        if((i % printevery) ==0 && mpirank==0) cout << "Adapt iteration " << i << endl;
        
#ifdef _OPENMPI
        // Model Mixing step
        if(mpirank==0){axb.drawvec(gen);} else {axb.drawvec_mpislave(gen);}

        // Get the current model mixing weights    
        wts_iter = mxd::Zero(nummodels+1,nvec[0]); //resets wt matrix
        axb.get_mix_wts(&diw, &wts_iter);  

        //Emulation Steps
        for(int j=0;j<nummodels;j++){
            for(int k=0;k<nummodels;k++){
                // Get the mixed prediction
                if(k!=j){
                    for(int l=0;l<nvec[0];l++){
                        mixprednotj[l] = mixprednotj[l] + fi(l,k+1)*wts_iter(k+1,l);  
                    }
                }
            }

            // add the discrepancy and get the weighted field obs
            for(int l=0;l<nvec[0];l++){
                mixprednotj[l] = mixprednotj[l] + wts_iter(0,l);
                y_list[j+1][nvec[j+1] + l] = (y_list[0][l] - mixprednotj[l])/wts_iter(j+1,l);
            }

            // Update emulator
            if(mpirank==0){ambm_list[j].drawvec(gen);} else {ambm_list[j].drawvec_mpislave(gen);}

            //Update finfo column
            ambm_list[j].predict(&dimix_list[j]);
            for(int l=0;l<nvec[0];l++){
                fi(l,j+1) = fmix_list[j][l] + means_list[j+1]; 
            }

        } 
        
#else
        // Mixing Draw
        axb.drawvec(gen);

        // Get the current model mixing weights    
        wts_iter = mxd::Zero(nummodels+1,nvec[0]); //resets wt matrix
        axb.get_mix_wts(&diw, &wts_iter);  

        //Emulation Steps
        for(int j=0;j<nummodels;j++){
            for(int k=0;k<nummodels;k++){
                // Get the mixed prediction
                if(k!=j){
                    for(int l=0;l<nvec[0];l++){
                        mixprednotj[l] = mixprednotj[l] + fi(l,k+1)*wts_iter(k+1,l);  
                    }
                }
            }
        
            // add the discrepancy and get the weighted field obs
            for(int l=0;l<nvec[0];l++){
                mixprednotj[l] = mixprednotj[l] + wts_iter(0,l);
                y_list[j+1][nvec[j+1] + l] = (y_list[0][l] - mixprednotj[l])/wts_iter(j+1,l);
            }
            ambm_list[j].drawvec(gen);
            
            //Update finfo column
            ambm_list[j].predict(&dimix_list[j]);
            for(int l=0;l<nvec[0];l++){
                fi(l,j+1) = fmix_list[j][l] + means_list[j+1]; 
            }
        }
#endif
    }

    //-------------------------------------------------- 
    // Cleanup.
#ifdef _OPENMPI
    //delete[] lwr_vec; //make pointer friendly
    //delete[] upr_vec; //make pointer friendly
    MPI_Finalize();
#endif
    return 0;
}