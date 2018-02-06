#include <iostream>
#include <cmath>

template<typename number>
number square(const number x)
{
  return x*x;
}

template<int N>
struct Point
{
    int elements[N];
};


template <int N>
double norm (const Point<N> &v)
{
 double tmp = 0;
 for (int i=0; i<N; ++i) 
   tmp += square(v.elements[i]);
 return std::sqrt(tmp);
};

int main()
{
  Point<2> p;
  p.elements[0]=1.0;
  p.elements[1]=4.0;
  
  std::cout << norm (p) << std::endl;  
}
