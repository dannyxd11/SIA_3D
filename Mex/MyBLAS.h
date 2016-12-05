#ifndef MyBLAS_H
#define MyBLAS_H

#include "my_types.h"
#include "blas.h"

void MatrixVectorBLAS(double *matrix, double *vector, double *returnVector, unsigned long m, unsigned long n, char chn);
double InnerProductBLAS(double *v1, double *v2,  unsigned long m, unsigned long incx, unsigned long incy);
void AddVectorsBLAS(double *v1, double *v2, double alpha, unsigned long m, unsigned long incx, unsigned long incy);
void ScaleVectorBLAS(double *v1, double alpha, unsigned long szVector);
void OuterBLAS(double *m1, double *v1, double *v2, double alpha, unsigned long m, unsigned long n);
double CalculateNormBLAS(double *atom, unsigned long szVector, unsigned long incx);
void MatrixMultiplyBLAS(double *m1, double *m2, double *mReturn, unsigned long m, unsigned long n, unsigned long k, char chn1, char chn2);
void PlaneRotateBLAS( double *DA, double *DB, double *DC, double *DS);
void SwapElementsBLAS(double *swapA, double *swapB, unsigned long nRows);
unsigned long MaxAbsBLAS(double *v, unsigned long m);
unsigned long MaxAbsDiagBLAS(double *v, ulng nRows);


/* 
 * Start of BLAS functions
 */

void SwapElementsBLAS(double *swapA, double *swapB, long nRows)
{
    ptrdiff_t incx = 1;
    ptrdiff_t nRows_t = static_cast<ptrdiff_t>(nRows);
    dswap(&nRows_t, swapA, &incx, swapB, &incx);        
}

double CalculateNormBLAS(double *atom, unsigned long szVector, unsigned long incx = 1)
{
    ptrdiff_t t1 = static_cast<ptrdiff_t>(szVector);
    ptrdiff_t incx_t = static_cast<ptrdiff_t>(incx);
    return dnrm2(&t1,atom,&incx_t);
}

void OuterBLAS(double *m1, double *v1, double *v2, double alpha, unsigned long m, unsigned long n)
{
    /* Remember n is the number of columns in the matrix so it wil be one more than the index in 
     * an array of the same size
     */
    ptrdiff_t incx = 1;
    ptrdiff_t t1,t2;
    t1 = static_cast<ptrdiff_t>(m);
    t2 = static_cast<ptrdiff_t>(n);
    dger(&t1,&t2,&alpha,v1,&incx,v2,&incx,m1,&t1);
}


void ScaleVectorBLAS(double *v1, double alpha, unsigned long szVector)
{
    ptrdiff_t incx = 1;
    ptrdiff_t t1 = static_cast<ptrdiff_t>(szVector);
    dscal(&t1, &alpha, v1, &incx);   
}


void MatrixVectorBLAS(double *matrix, double *vector, double *returnVector, 
        unsigned long m, unsigned long n, char chn)
{
    /* Remember n is the number of columns in the matrix so it wil be one more than the index in 
     * an array of the same size
     */
    double zero = 0, one = 1;
    ptrdiff_t incx = 1;
    ptrdiff_t t1,t2;
    t1 = static_cast<ptrdiff_t>(m);
    t2 = static_cast<ptrdiff_t>(n);
    dgemv(&chn, &t1, &t2, &one, matrix, &t1, vector, &incx, &zero, returnVector, &incx);
}

//void MatrixMultiplyBLAS(double *m1, double *m2, double *mReturn, unsigned long m, unsigned long n, unsigned long k, char chn1, char chn2)
//{
//    /* Remember n is the number of columns in the matrix so it wil be one more than the index in 
//     * an array of the same size
//     */
//	// m is the number of rows of m1
//	// n is the number of columns of m1
//	// k is the number of columns of mReturn
//    ptrdiff_t t1,t2,t3, LDA, LDB, LDC;
//    LDC = m;
//    if (chn1 == 'T')
//    {
//        LDA = n;
//        LDB = n;
//    }else{
//        LDA = m;
//        LDB = n;
//    }
//
//    double zero = 0, one = 1;
//    t1 = static_cast<ptrdiff_t>(m);
//    t2 = static_cast<ptrdiff_t>(n);
//    t3 = static_cast<ptrdiff_t>(k);
//    dgemm(&chn1,&chn2, &t1, &t3, &t2, &one, m1, &LDA, m2, &LDB, &zero, mReturn, &LDC);
//     
//}

void MatrixMultiplyBLAS(double *m1, double *m2, double *mReturn, unsigned long m, unsigned long n, unsigned long k, char chn1, char chn2)
{
    /* Remember n is the number of columns in the matrix so it wil be one more than the index in 
     * an array of the same size
     */
	// m is the number of rows of m1
	// n is the number of columns of m1
	// k is the number of columns of mReturn


	/* Always fixed */
	// The dimensions of m1 are an input to the function it is always m*n
	ptrdiff_t LDA = m;
	//ptrdiff_t LDB = n;
	// The number of columns of mReturn is an input to the function
	ptrdiff_t n_t = k;

	ptrdiff_t m_t, k_t;
	if (chn1 == 'T')
	{
		// If 1st matrix is not transposed then dimensions of m1 and op(m1) are switched
		m_t = n;
		k_t = m;
	}else{
		// If 1st matrix is not transposed then dimensions of m1 and op(m1) are the same
		m_t = m;
		k_t = n;
	}

	// If transposed then 1st dim is 2nd dim of return vector otherwise has to be number of columns in op(m1)
	ptrdiff_t LDB;
	if (chn2 == 'T')
	{
		LDB = k;
	}else{
		LDB = k_t;
	}

	// The 1st dimension of the return matrix is the same as the 1st dimension of op(m1)
	ptrdiff_t LDC = m_t;
    double zero = 0, one = 1;

    dgemm(&chn1,&chn2, &m_t, &n_t, &k_t, &one, m1, &LDA, m2, &LDB, &zero, mReturn, &LDC);
}

double InnerProductBLAS(double *v1, double *v2,  unsigned long m, unsigned long incx = 1, unsigned long incy = 1)
{
    ptrdiff_t t1 = static_cast<ptrdiff_t>(m);
    ptrdiff_t incx_t = static_cast<ptrdiff_t>(incx);
    ptrdiff_t incy_t = static_cast<ptrdiff_t>(incy);
    return ddot(&t1, v1, &incx_t, v2, &incy_t);    
}

void AddVectorsBLAS(double *v1, double *v2, double alpha, unsigned long m, unsigned long incx, unsigned long incy )
{
    ptrdiff_t incx_t = static_cast<ptrdiff_t>(incx);
    ptrdiff_t incy_t = static_cast<ptrdiff_t>(incy);
    ptrdiff_t t1 = static_cast<ptrdiff_t>(m);
    daxpy(&t1, &alpha, v1, &incx_t, v2, &incy_t);
}

void PlaneRotateBLAS( double *DA, double *DB, double *DC, double *DS)
{
   drotg(DA,DB,DC,DS);    
}

void ApplyPlaneRotateBLAS( double *v1, double *v2, unsigned long n, double *DC, double *DS)
{
    ptrdiff_t incx = 1;
    ptrdiff_t n_t = static_cast<ptrdiff_t>(n);
    drot(&n_t,v1,&incx,v2,&incx,DC,DS);
}

ulng MaxAbsBLAS(double *v, unsigned long m)
{
	ptrdiff_t incx = 1;
	ptrdiff_t m_t = static_cast<ptrdiff_t>(m);
	return static_cast<ulng>(idamax(&m_t, v, &incx));
}

ulng MaxAbsDiagBLAS(double *v, ulng nRows)
{
    // Distance between diagonl elements is nRows+1.
	ptrdiff_t incx = static_cast<ptrdiff_t>(nRows+1);
    // Assume matrix is square therefore number of elements to compare is 
    // equal to the number of rows.
	ptrdiff_t m_t = static_cast<ptrdiff_t>(nRows);
	return static_cast<ulng>(idamax(&m_t, v, &incx));
}


#endif
