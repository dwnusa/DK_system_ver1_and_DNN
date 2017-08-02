#include "mex.h"
#include "okFrontPanelDLL.h"
#include <string.h>
#include <stdio.h>

bool Transfer_pulse(int* dst,int* vector1,int* vector2,int* vector3,int* vector4, okCFrontPanel *dev, int numRows, int numCols, int numPS)
{
    //DAQ
    unsigned char *freebuffer1 = new unsigned char[numCols * sizeof(int)];
    unsigned char *freebuffer2 = new unsigned char[numCols * sizeof(int)];
    long ret1, ret2;
    
    //rawABCD ������ ȹ��
    dev->UpdateWireOuts();
    int tempWireOuts = (int)(dev->GetWireOutValue(0x20));
    int n2 = (int)((tempWireOuts & 0x00000004) >> 2);
    int n3 = (int)((tempWireOuts & 0x00000008) >> 3);
    if (n2 == 1)
        ret1 = dev->ReadFromBlockPipeOut(0xA2, numCols, numCols * sizeof(int), freebuffer1); //rawABCD ������ ȹ��
    else
        return false;
    if (n3 == 1)
        ret2 = dev->ReadFromBlockPipeOut(0xA3, numCols, numCols * sizeof(int), freebuffer2); //rawABCD ������ ȹ��
    else
        return false;
    
    int current_idx = 0;
    int last_idx = -1;
    int idx = 0;
    for (int i = 0; i < (numCols) * sizeof(int); i += sizeof(int))
    { 
        int x = i / sizeof(int);
        
        //���� ���ڵ� (ä�� : A,B) from freebuffer1
        int temp_A = (short)(((freebuffer1[3 + i] << 8) | (freebuffer1[2 + i])));// >> 2);
        int temp_B = (short)(((freebuffer1[1 + i] << 8) | (freebuffer1[0 + i])));// >> 2);
        
        //���� ���ڵ� (ä�� : C,D) from freebuffer2
        int temp_C = (short)(((freebuffer2[3 + i] << 8) | (freebuffer2[2 + i])));// >> 2);
        int temp_D = (short)(((freebuffer2[1 + i] << 8) | (freebuffer2[0 + i])));// >> 2);
        
        int idxA = temp_A & 0x00000003;
        int idxB = temp_B & 0x00000003;
        int idxC = temp_C & 0x00000003;
        int idxD = temp_D & 0x00000003;
        if ((idxA == idxB) && (idxB == idxC) && (idxC == idxD))
        {
            current_idx = idxA;
            if (current_idx != last_idx) //
            {
                idx = x; //������ index�� ����
                last_idx = current_idx; //����(t) �ε����� ����(t-1) �ε����� ����
            }
            else //index(t-1)�� index(t)�� ���� ������
            {
                idx = idx; //���� index�� ��� ���� (���� index�� �����ϰԵ�)
            }

            dst[x*numRows + 0] = (int)((int)temp_A >> 2);
            dst[x*numRows + 1] = (int)((int)temp_B >> 2);
            dst[x*numRows + 2] = (int)((int)temp_C >> 2);
            dst[x*numRows + 3] = (int)((int)temp_D >> 2);
            dst[x*numRows + 4] = (int)idx;
        }
        else
        {
            current_idx = 0;
            last_idx = -1;
            idx = 0;
            mexPrintf("index missmatched\n");
        }
    }
	
    int x = 0;
    int y = 0;
    while ((x < numCols) && ((x+numPS) < numCols))
    { 
        int index = dst[x*numRows + 4]; //������ index�� ����
        int indexAfter_numPS = dst[(x+numPS-1)*numRows + 4]; //(numPS)��°���� index�� ����
        if (index == indexAfter_numPS) //(numPS)��° �տ� �ִ� index�� ���ؼ� ��ġ�ϸ�
        {
            for(int i = 0; i < numPS; i++) //vector1,2,3,4�� 
            {
                vector1[y*numPS+i] = dst[(x+i)*numRows + 0];
                vector2[y*numPS+i] = dst[(x+i)*numRows + 1];
                vector3[y*numPS+i] = dst[(x+i)*numRows + 2];
                vector4[y*numPS+i] = dst[(x+i)*numRows + 3];
            }
            x += numPS;
            y++;
        }
        else //100��° �տ� �ִ� index�� ���ؼ� ��ġ���� ������
        {
            x = indexAfter_numPS;
        }
    }
    //�޸� ����
    delete [] freebuffer1;
    delete [] freebuffer2;
    
    return true;
}

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    //Matlab Input Matrix Size
    if(nrhs != 4)
        mexErrMsgTxt("Invalid number of input arguments");
    if((nlhs != 1) && (nlhs != 5))
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

	int* numPS = (int*)mxGetData(prhs[3]);
    
    if(numRows[0] != 5)
		mexErrMsgTxt("Invalid buffer size. It must be 5x(buffer)");
    
    plhs[0] = mxCreateNumericMatrix(numRows[0], numCols[0], mxINT32_CLASS, mxREAL);
    int* out = (int*)mxGetData(plhs[0]);
	int* pulse_vector1;
	int* pulse_vector2;
	int* pulse_vector3;
	int* pulse_vector4;
    if(nlhs == 5)
	{
        int numVectors = numCols[0]/numPS[0];
		plhs[1] = mxCreateNumericMatrix(numPS[0], numVectors, mxINT32_CLASS, mxREAL);
		plhs[2] = mxCreateNumericMatrix(numPS[0], numVectors, mxINT32_CLASS, mxREAL);
		plhs[3] = mxCreateNumericMatrix(numPS[0], numVectors, mxINT32_CLASS, mxREAL);
		plhs[4] = mxCreateNumericMatrix(numPS[0], numVectors, mxINT32_CLASS, mxREAL);
		pulse_vector1 = (int*)mxGetData(plhs[1]);
		pulse_vector2 = (int*)mxGetData(plhs[2]);
		pulse_vector3 = (int*)mxGetData(plhs[3]);
		pulse_vector4 = (int*)mxGetData(plhs[4]);
	}
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
    
    unsigned int temp = ep00wire[0] & 0xFFBFFFFF; //wren�� �����Ѵ�. bit22(0)
    dev->SetWireInValue( (int)0x00, (unsigned int)temp, (unsigned int)0xffffffff );
    dev->UpdateWireIns();
    
    bool result = Transfer_pulse(out,pulse_vector1,pulse_vector2,pulse_vector3,pulse_vector4, dev, numRows[0], numCols[0], numPS[0]);
    
    temp = ep00wire[0] | 0x00400000; //wren�� �����Ѵ�. bit22(1)
    dev->SetWireInValue( (int)0x00, (unsigned int)temp, (unsigned int)0xffffffff );
    dev->UpdateWireIns();
    
    //����
    dev->~okCFrontPanel();
}
