#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <thread>
#include <chrono>

#include <mpi.h>

void broadcast_demo(int rank, int size)
{
    int data[2];
    
    {
      srand(rank+time(NULL));
      data[0] = rand()%100;
      data[1] = rand()%100;
    }

    std::cout << " I am rank " << rank << " and my data was " << data[0] << " " << data[1] << std::endl;

    int root = 0;
    MPI_Bcast(&data, 2, MPI_INT, root, MPI_COMM_WORLD);
  
  std::cout << " I am rank " << rank << " and my data is  " << data[0] << " " << data[1] << std::endl;
}

void allreduce_demo(int rank, int size)
  
{ 
  double my_number = 1.0 + rank/10.0;
  std::cout << " I am rank " << rank << " and my number is  " << my_number << std::endl;
  
  double result = 0.0;
  MPI_Allreduce(&my_number, &result, 1, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
  
  std::cout << " I am rank " << rank << " and result is " << result << std::endl;
}

  

int main(int argc, char **argv)
{
  MPI_Init(&argc, &argv);

  int rank;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  broadcast_demo(rank, size);

  MPI_Barrier(MPI_COMM_WORLD);
  
  allreduce_demo(rank, size);  
        
  MPI_Finalize();
}
