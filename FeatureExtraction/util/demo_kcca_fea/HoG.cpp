#include <stdlib.h>
#include <math.h>
#include <mex.h>
#include <vector>

using namespace std;   

void HoG(double *pixels, double *params, int *img_size, double *dth_des, unsigned int grayscale){
    
    const float pi = 3.1415926536;
    
    int nb_bins       = (int) params[0];
    double cwidth     =  params[1];
    int block_size    = (int) params[2];
    int orient        = (int) params[3];
    double clip_val   = params[4];
    
    int img_width  = img_size[1];
    int img_height = img_size[0];
    
    int hist1= 2+ceil(-0.5 + img_height/cwidth);
    int hist2= 2+ceil(-0.5 + img_width/cwidth);
    
    double bin_size = (1+(orient==1))*pi/nb_bins;
    
    float dx[3], dy[3], grad_or, grad_mag, temp_mag;
    float Xc, Yc, Oc, block_norm;
    int x1, x2, y1, y2, bin1, bin2;
    int des_indx = 0;
    
    vector<vector<vector<double> > > h(hist1, vector<vector<double> > (hist2, vector<double> (nb_bins, 0.0) ) );    
    vector<vector<vector<double> > > block(block_size, vector<vector<double> > (block_size, vector<double> (nb_bins, 0.0) ) );
    
    //Calculate gradients (zero padding)
    
    for(unsigned int y=0; y<img_height; y++) {
        for(unsigned int x=0; x<img_width; x++) {
            if (grayscale == 1){
                if(x==0) dx[0] = pixels[y +(x+1)*img_height];
                else{
                    if (x==img_width-1) dx[0] = -pixels[y + (x-1)*img_height];
                    else dx[0] = pixels[y+(x+1)*img_height] - pixels[y + (x-1)*img_height];
                }
                if(y==0) dy[0] = -pixels[y+1+x*img_height];
                else{
                    if (y==img_height-1) dy[0] = pixels[y-1+x*img_height];
                    else dy[0] = -pixels[y+1+x*img_height] + pixels[y-1+x*img_height];
                }
            }
            else{
                if(x==0){
                    dx[0] = pixels[y +(x+1)*img_height];
                    dx[1] = pixels[y +(x+1)*img_height + img_height*img_width];
                    dx[2] = pixels[y +(x+1)*img_height + 2*img_height*img_width];                    
                }
                else{
                    if (x==img_width-1){
                        dx[0] = -pixels[y + (x-1)*img_height];                        
                        dx[1] = -pixels[y + (x-1)*img_height + img_height*img_width];
                        dx[2] = -pixels[y + (x-1)*img_height + 2*img_height*img_width];
                    }
                    else{
                        dx[0] = pixels[y+(x+1)*img_height] - pixels[y + (x-1)*img_height];
                        dx[1] = pixels[y+(x+1)*img_height + img_height*img_width] - pixels[y + (x-1)*img_height + img_height*img_width];
                        dx[2] = pixels[y+(x+1)*img_height + 2*img_height*img_width] - pixels[y + (x-1)*img_height + 2*img_height*img_width];
                        
                    }
                }
                if(y==0){
                    dy[0] = -pixels[y+1+x*img_height];
                    dy[1] = -pixels[y+1+x*img_height + img_height*img_width];
                    dy[2] = -pixels[y+1+x*img_height + 2*img_height*img_width];
                }
                else{
                    if (y==img_height-1){
                        dy[0] = pixels[y-1+x*img_height];
                        dy[1] = pixels[y-1+x*img_height + img_height*img_width];
                        dy[2] = pixels[y-1+x*img_height + 2*img_height*img_width];
                    }
                    else{
                        dy[0] = -pixels[y+1+x*img_height] + pixels[y-1+x*img_height];
                        dy[1] = -pixels[y+1+x*img_height + img_height*img_width] + pixels[y-1+x*img_height + img_height*img_width];
                        dy[2] = -pixels[y+1+x*img_height + 2*img_height*img_width] + pixels[y-1+x*img_height + 2*img_height*img_width];
                    }
                }
            }
            
            grad_mag = sqrt(dx[0]*dx[0] + dy[0]*dy[0]);
            grad_or= atan2(dy[0], dx[0]);
            
            if (grayscale == 0){
                temp_mag = grad_mag;
                for (unsigned int cli=1;cli<3;++cli){
                    temp_mag= sqrt(dx[cli]*dx[cli] + dy[cli]*dy[cli]);
                    if (temp_mag>grad_mag){
                        grad_mag=temp_mag;
                        grad_or= atan2(dy[cli], dx[cli]);
                    }
                }
            }
            
            if (grad_or<0) grad_or+=pi + (orient==1) * pi;

            // trilinear interpolation
            
            bin1 = (int)floor(0.5 + grad_or/bin_size) - 1;
            bin2 = bin1 + 1;
            x1   = (int)floor(0.5+ x/cwidth);
            x2   = x1+1;
            y1   = (int)floor(0.5+ y/cwidth);
            y2   = y1 + 1;
            
            Xc = (x1+1-1.5)*cwidth + 0.5;
            Yc = (y1+1-1.5)*cwidth + 0.5;
            
            Oc = (bin1+1+1-1.5)*bin_size;
            
            if (bin2==nb_bins){
                bin2=0;
            }
            if (bin1<0){
                bin1=nb_bins-1;
            }            
           
            h[y1][x1][bin1]= h[y1][x1][bin1]+grad_mag*(1-((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
            h[y1][x1][bin2]= h[y1][x1][bin2]+grad_mag*(1-((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
            h[y2][x1][bin1]= h[y2][x1][bin1]+grad_mag*(1-((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
            h[y2][x1][bin2]= h[y2][x1][bin2]+grad_mag*(1-((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
            h[y1][x2][bin1]= h[y1][x2][bin1]+grad_mag*(((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
            h[y1][x2][bin2]= h[y1][x2][bin2]+grad_mag*(((x+1-Xc)/cwidth))*(1-((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
            h[y2][x2][bin1]= h[y2][x2][bin1]+grad_mag*(((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(1-((grad_or-Oc)/bin_size));
            h[y2][x2][bin2]= h[y2][x2][bin2]+grad_mag*(((x+1-Xc)/cwidth))*(((y+1-Yc)/cwidth))*(((grad_or-Oc)/bin_size));
        }
    }
    
    
    
    //Block normalization
    
    for(unsigned int x=1; x<hist2-block_size; x++){
        for (unsigned int y=1; y<hist1-block_size; y++){
            
            block_norm=0;
            for (unsigned int i=0; i<block_size; i++){
                for(unsigned int j=0; j<block_size; j++){
                    for(unsigned int k=0; k<nb_bins; k++){
                        block_norm+=h[y+i][x+j][k]*h[y+i][x+j][k];
                    }
                }
            }
            
            block_norm=sqrt(block_norm);
            for (unsigned int i=0; i<block_size; i++){
                for(unsigned int j=0; j<block_size; j++){
                    for(unsigned int k=0; k<nb_bins; k++){
                        if (block_norm>0){
                            block[i][j][k]=h[y+i][x+j][k]/block_norm;
                            if (block[i][j][k]>clip_val) block[i][j][k]=clip_val;
                        }
                    }
                }
            }
            
            block_norm=0;
            for (unsigned int i=0; i<block_size; i++){
                for(unsigned int j=0; j<block_size; j++){
                    for(unsigned int k=0; k<nb_bins; k++){
                        block_norm+=block[i][j][k]*block[i][j][k];
                    }
                }
            }
            
            block_norm=sqrt(block_norm);
            for (unsigned int i=0; i<block_size; i++){
                for(unsigned int j=0; j<block_size; j++){
                    for(unsigned int k=0; k<nb_bins; k++){
                        if (block_norm>0) dth_des[des_indx]=block[i][j][k]/block_norm;
                        else dth_des[des_indx]=0.0;
                        des_indx++;
                    }
                }
            }
        }
    }
}


void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[]) {
    
    double *pixels, *dth_des, *params;
    int nb_bins, block_size;
    int img_size[2];
    unsigned int grayscale = 1;
    
    if (nlhs>1)  mexErrMsgTxt("Too many output arguments");
    if (nrhs==0) mexErrMsgTxt("No Image -> No HoG");
    
    if (mxGetClassID(prhs[0])!=6) mexErrMsgTxt("Matrix is not of type double");
    
    pixels     = mxGetPr(prhs[0]);    
    
    img_size[0] = mxGetM(prhs[0]);
    img_size[1]  = mxGetN(prhs[0]);
    if (mxGetNumberOfDimensions(prhs[0])==3){
        img_size[1] /= 3;
        grayscale = 0;
    }
    
    if (nrhs>1){
        params     = mxGetPr(prhs[1]);
        if (params[0]<=0) mexErrMsgTxt("Number of orientation bins must be positive");
        if (params[1]<=0) mexErrMsgTxt("Cell size must be positive");
        if (params[2]<=0) mexErrMsgTxt("Block size must be positive");
    }
    else {
        params = new double[5];
        params[0]=9;
        params[1]=8;
        params[2]=2;
        params[3]=0;
        params[4]=0.2;
    }
    
    nb_bins       = (int) params[0];    
    block_size    = (int) params[2];     
    
    int hist1= 2+ceil(-0.5 + img_size[0]/params[1]);
    int hist2= 2+ceil(-0.5 + img_size[1]/params[1]);

    plhs[0] = mxCreateDoubleMatrix((hist1-2-(block_size-1))*(hist2-2-(block_size-1))*nb_bins*block_size*block_size, 1, mxREAL);
    dth_des = mxGetPr(plhs[0]);
    
    HoG(pixels, params, img_size, dth_des, grayscale);
    if (nrhs==1) delete[] params;
}
