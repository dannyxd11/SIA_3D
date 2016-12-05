#ifndef MyLAPACK_H
#define MyLAPACK_H

#include "lapack.h"
#include <algorithm>
using namespace std;
void SVDLapack(double *mIn, double *mSingular, unsigned long m, unsigned long n)
{
    double *u = new double[1];
    double *v = new double[1];
	ptrdiff_t dummy = 1;
	
	// Determine size of work as greater than MAX(1,3*MIN(M,N)+MAX(M,N),5*MIN(M,N))
	//ptrdiff_t lwork = max(3*min(m,n)+max(m,n),5*min(m,n));
	ptrdiff_t lwork = static_cast<ptrdiff_t>(max(static_cast<unsigned long>(1),max(3*min(m,n)+max(m,n),5*min(m,n)))+10);
    ptrdiff_t  info;
    double *work = new double[lwork];
    char jobu = 'N';
    char jobvt = 'N';   
    ptrdiff_t m_ptrdiff_t = static_cast<ptrdiff_t>(m);
    ptrdiff_t n_ptrdiff_t = static_cast<ptrdiff_t>(n);
    //dgesvd( &jobu, &jobvt,&m_ptrdiff_t, &n_ptrdiff_t, mIn, &m_ptrdiff_t, mSingular, u, &m_ptrdiff_t, v, &n_ptrdiff_t,work,&lwork,&info);
	dgesvd( &jobu, &jobvt,&m_ptrdiff_t, &n_ptrdiff_t, mIn, &m_ptrdiff_t, mSingular, u, &dummy, v, &dummy,work,&lwork,&info);

	delete[] u;
    delete[] v;
    delete[] work;
	v = 0;
	u = 0;
	work = 0;
}




#endif
