#include "mex.h"
#include "okFrontPanelDLL.h"
#include <string.h>
#include <stdio.h>

#define MIN(a,b)   (((a)<(b)) ? (a) : (b))

bool Transfer_realtime(int* dst, okCFrontPanel *dev, int numRows, int numCols)
{
    //DAQ
    unsigned char *freebuffer0 = new unsigned char[numCols * sizeof(int)];

    long ret0, ret3;
    
    ret0 = dev->ReadFromBlockPipeOut(0xA0, numCols, numCols * sizeof(int), freebuffer0);
    
    if (ret0 < 0) 
    {
        switch (ret0) {
            case okCFrontPanel::InvalidBlockSize:
                mexPrintf("Block Size Not Supported(0)\n");
                break;
            case okCFrontPanel::UnsupportedFeature:
                mexPrintf("Unsupported Feature(0)\n");
                break;
            default:
                mexPrintf("Transfer_realtime Failed with error(0): %ld\n", ret0);
                break;
        }

        if (dev->IsOpen() == false){
            mexPrintf("Device disconnected(0)\n");
        } 
        
        return false;
	}
    
    for (int i = 0; i <= (numCols-1) * sizeof(int); i += sizeof(int))
    { 
        int x = i / 4;
        
        //(X,Y,SUM) 디코딩
        unsigned int temp_XYSUM = (freebuffer0[3 + i] << 24) | (freebuffer0[2 + i] << 16) | (freebuffer0[1 + i] << 8) | (freebuffer0[0 + i]);

        int valueX = (unsigned int)((temp_XYSUM & 0xFF800000) >> 23);//X좌표 9bit
        int valueY = (unsigned int)((temp_XYSUM & 0x007FC000) >> 14);//Y좌표 9bit
        int valueSUM = (unsigned int)((temp_XYSUM & 0x00003FFF));//Sum값 14bit

        dst[x*numRows + 0] = (int)valueX;
        dst[x*numRows + 1] = (int)valueY;
        dst[x*numRows + 2] = (int)valueSUM;
    }
	
    //메모리 해제
	delete [] freebuffer0;    
    return true;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //Matlab Input Matrix Size
    if(nrhs != 3)
        mexErrMsgTxt("Invalid number of input arguments");
    if(nlhs != 1)
        mexErrMsgTxt("Invalid number of outputs");
    if(!mxIsInt32(prhs[0]))
        mexErrMsgTxt("input buffer type must be single");
    if(!mxIsInt32(prhs[1]))
        mexErrMsgTxt("input buffer type must be single");
    if(!mxIsUint32(prhs[2]))
        mexErrMsgTxt("input ep00wire type must be uint32");
        
    int* numRows = (int*)mxGetData(prhs[0]);
    int* numCols = (int*)mxGetData(prhs[1]);
    
    unsigned int* ep00wire = (unsigned int*)mxGetData(prhs[2]);
    
    if(numRows[0] != 3)// || (numCols != 1024))
		mexErrMsgTxt("Invalid buffer size. It must be 3x(buffer)");
       
    //Opalkelly variable
	okCFrontPanel *dev = new okCFrontPanel;
    
    dev->OpenBySerial("");//14230007CO
    
    if (dev->IsOpen()) {} //mexPrintf("IsOpen Pass\n");
    else{
        mexPrintf("IsOpen Fail\n");
        dev->~okCFrontPanel();
        return;}
    if (dev->IsFrontPanelEnabled()) {}//mexPrintf("FrontPanel support is enabled.\n");
    else{
        mexPrintf("FrontPanel support is not enabled.\n");
        dev->~okCFrontPanel();
        return;}
    
    dev->SetWireInValue( (int)0x00, (unsigned int)ep00wire[0], (unsigned int)0xffffffff );
    dev->UpdateWireIns();
    
    //fifo 상태확인용
    dev->UpdateWireOuts();
    int tempWireOuts = (int)(dev->GetWireOutValue(0x20));
    //(X,Y,SUM) 상태확인
    int n0 = (int)((tempWireOuts & 0x00000001) >> 0);
    //(featureB) 상태확인
    //int n1 = (int)((tempWireOuts & 0x00000002) >> 1);
    
    if (n0 == 1)
    {
        plhs[0] = mxCreateNumericMatrix(numRows[0], numCols[0], mxINT32_CLASS, mxREAL);
        int* out = (int*)mxGetData(plhs[0]);
        bool verify = Transfer_realtime(out, dev, numRows[0], numCols[0]);
        if (verify == false)
        {
            plhs[0] = mxCreateNumericMatrix(numRows[0], 1, mxINT32_CLASS, mxREAL);
            out = (int*)mxGetData(plhs[0]);
        }
    }
    else
    {
        plhs[0] = mxCreateNumericMatrix(numRows[0], 1, mxINT32_CLASS, mxREAL);
        int* out = (int*)mxGetData(plhs[0]);
    }
    //종료
    dev->~okCFrontPanel();
}
