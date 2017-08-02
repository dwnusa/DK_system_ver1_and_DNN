#include "mex.h"
#include "matrix.h"
#include "gpu_find_index.h"
#include <string.h>
#include <stdio.h>
//mex -g helloMex.cpp;
void mex_find_index(bool* dst, double* origX, double* origY, double* thrX, double* thrY, int numRows_orig, int numRows_thr)
{
    for(int r = 0; r < numRows_orig; r++)
    {
        for(int c = 0; c < numRows_thr; c++)
        {
            if (origX[r] == thrX[c])
            {
                if (origY[r] == thrY[c])
                {
                    dst[r] = true;
                }
            }
        }
    }
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //mexPrintf("Hello, mex!\n");
	
    //Matlab Input Matrix Size
    if(nrhs != 4)
        mexErrMsgTxt("Invalid number of input arguments");
    if(nlhs != 1)
        mexErrMsgTxt("Invalid number of outputs");
    if(!mxIsDouble(prhs[0]))
        mexErrMsgTxt("input origX type must be double");
    if(!mxIsDouble(prhs[1]))
        mexErrMsgTxt("input origY type must be double");
    if(!mxIsDouble(prhs[2]))
        mexErrMsgTxt("input thrX type must be double");
    if(!mxIsDouble(prhs[3]))
        mexErrMsgTxt("input thrY type must be double");
    int numRows_origX = (int)mxGetM(prhs[0]);
    int numCols_origX = (int)mxGetN(prhs[0]);
    int numRows_origY = (int)mxGetM(prhs[1]);
    int numCols_origY = (int)mxGetN(prhs[1]);
    int numRows_thrX = (int)mxGetM(prhs[2]);
    int numCols_thrX = (int)mxGetN(prhs[2]);
    int numRows_thrY = (int)mxGetM(prhs[3]);
    int numCols_thrY = (int)mxGetN(prhs[3]);
    
    if((numCols_origX != 1))// || ((numCols % 64) != 0))
		mexErrMsgTxt("Invalid array numCols_origX size. It must be 1");
    if((numCols_origY != 1))// || ((numCols % 64) != 0))
		mexErrMsgTxt("Invalid array numCols_origY size. It must be 1");
    if((numCols_thrX != 1))// || ((numCols % 64) != 0))
		mexErrMsgTxt("Invalid array numCols_thrX size. It must be 1");
    if((numCols_thrY != 1))// || ((numCols % 64) != 0))
		mexErrMsgTxt("Invalid array numCols_thrY size. It must be 1");
    
    double* origX = (double*)mxGetData(prhs[0]);
    double* origY = (double*)mxGetData(prhs[1]);
    double* thrX = (double*)mxGetData(prhs[2]);
    double* thrY = (double*)mxGetData(prhs[3]);
    
    plhs[0] = mxCreateNumericMatrix(numRows_origX, 1, mxLOGICAL_CLASS, mxREAL);
    bool* out = (bool*)mxGetData(plhs[0]);
	memset(out, false, sizeof(bool) * numRows_origX);
    
    //mex_find_index(out, origX, origY, thrX, thrY, numRows_origX, numRows_thrX); //c++ 코드
    gpu_find_index(out, origX, origY, thrX, thrY, numRows_origX, numRows_thrX); //CUDA 코드 "gpu_find_index.h"
    
    mexPrintf("Success, mex!\n");
}