#include <deal.II/base/quadrature_lib.h>
#include <deal.II/base/function.h>
#include <deal.II/base/logstream.h>
#include <deal.II/lac/vector.h>
#include <deal.II/lac/full_matrix.h>
#include <deal.II/lac/sparse_matrix.h>
#include <deal.II/lac/dynamic_sparsity_pattern.h>
#include <deal.II/lac/solver_cg.h>
#include <deal.II/lac/precondition.h>
#include <deal.II/grid/tria.h>
#include <deal.II/dofs/dof_handler.h>
#include <deal.II/grid/grid_generator.h>
#include <deal.II/grid/tria_accessor.h>
#include <deal.II/grid/tria_iterator.h>
#include <deal.II/grid/manifold_lib.h>
#include <deal.II/dofs/dof_accessor.h>
#include <deal.II/dofs/dof_tools.h>
#include <deal.II/fe/fe_values.h>
#include <deal.II/numerics/vector_tools.h>
#include <deal.II/numerics/matrix_tools.h>
#include <deal.II/numerics/data_out.h>
#include <deal.II/base/function_lib.h>

#include <deal.II/meshworker/dof_info.h>
#include <deal.II/meshworker/integration_info.h>
#include <deal.II/meshworker/assembler.h>
#include <deal.II/meshworker/loop.h>

#include <fstream>
#include <iostream>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>


#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

using namespace dealii;



template <int dim>
class Estimator : public MeshWorker::LocalIntegrator<dim>
{
public:
  void cell(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const;
  void boundary(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const;
  void face(MeshWorker::DoFInfo<dim> &dinfo1,
            MeshWorker::DoFInfo<dim> &dinfo2,
            typename MeshWorker::IntegrationInfo<dim> &info1,
            typename MeshWorker::IntegrationInfo<dim> &info2) const;
};


template <int dim>
void Estimator<dim>::cell(MeshWorker::DoFInfo<dim> &dinfo, typename MeshWorker::IntegrationInfo<dim> &info) const
{

  const FEValuesBase<dim> &fe = info.fe_values();

  const double h = dinfo.cell->diameter();

  const std::vector<Tensor<2,dim> > &DDuh = info.hessians[0][0];
  for (unsigned k=0; k<fe.n_quadrature_points; ++k)
    {
      const double t = trace(DDuh[k]);
      dinfo.value(0) +=  t*t*fe.JxW(k);
    }

  dinfo.value(0) = h*std::sqrt(dinfo.value(0)); // not h*h
}

template <int dim>
void Estimator<dim>::boundary(MeshWorker::DoFInfo<dim> &, typename MeshWorker::IntegrationInfo<dim> &) const
{}


template <int dim>
void Estimator<dim>::face(MeshWorker::DoFInfo<dim> &dinfo1,
                          MeshWorker::DoFInfo<dim> &dinfo2,
                          typename MeshWorker::IntegrationInfo<dim> &info1,
                          typename MeshWorker::IntegrationInfo<dim> &info2) const
{
  const FEValuesBase<dim> &fe = info1.fe_values();
  const std::vector<Tensor<1,dim> > &Duh1 = info1.gradients[0][0];
  const std::vector<Tensor<1,dim> > &Duh2 = info2.gradients[0][0];

  const double h = dinfo1.face->diameter();

  double result = 0.0;

  for (unsigned k=0; k<fe.n_quadrature_points; ++k)
    {
      double diff2 = fe.normal_vector(k) * Duh1[k] - fe.normal_vector(k) * Duh2[k];
      result += (h * diff2*diff2)*fe.JxW(k);
    }

  if (dinfo1.cell->is_locally_owned())
    dinfo1.value(0) = 0.5*std::sqrt(result);
  if (dinfo2.cell->is_locally_owned())
    dinfo2.value(0) = 0.5*std::sqrt(result);
}

template <int dim>
class Step6
{
public:
  Step6 (const int degree, const bool adaptive);
  ~Step6 ();

  void run ();

private:
  void setup_system ();
  void assemble_system ();
  void solve ();
  void refine_grid ();
  void output_results (const unsigned int cycle) const;

  const int            degree;
  const bool           adaptive;
  Triangulation<dim>   triangulation;

  DoFHandler<dim>      dof_handler;
  FE_Q<dim>            fe;

  ConstraintMatrix     constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;
  Vector<float>        estimate_kelly;

  BlockVector<double>  estimates;
};



template <int dim>
double coefficient (const Point<dim> &p)
{
  if (p.square() < 0.5*0.5)
    return 20;
  else
    return 1;
}





template <int dim>
Step6<dim>::Step6 (const int degree, const bool adaptive)
  :
  degree(degree),
  adaptive(adaptive),
  dof_handler (triangulation),
  fe (degree)
{}



template <int dim>
Step6<dim>::~Step6 ()
{
  dof_handler.clear ();
}



template <int dim>
void Step6<dim>::setup_system ()
{
  dof_handler.distribute_dofs (fe);

  solution.reinit (dof_handler.n_dofs());
  system_rhs.reinit (dof_handler.n_dofs());
  estimates.reinit(1);

  constraints.clear ();
  DoFTools::make_hanging_node_constraints (dof_handler,
                                           constraints);

  Functions::LSingularityFunction exact_solution;

  VectorTools::interpolate_boundary_values (dof_handler,
                                            0,
                                            exact_solution,
                                            constraints);


  constraints.close ();

  DynamicSparsityPattern dsp(dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);
}



template <int dim>
void Step6<dim>::assemble_system ()
{
  const QGauss<dim>  quadrature_formula(fe.degree+1);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
  cell = dof_handler.begin_active(),
  endc = dof_handler.end();
  for (; cell!=endc; ++cell)
    {
      cell_matrix = 0;
      cell_rhs = 0;

      fe_values.reinit (cell);

      for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
        {
          const double current_coefficient = 1.0;
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            {
              for (unsigned int j=0; j<dofs_per_cell; ++j)
                cell_matrix(i,j) += (current_coefficient *
                                     fe_values.shape_grad(i,q_index) *
                                     fe_values.shape_grad(j,q_index) *
                                     fe_values.JxW(q_index));

              cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                              0.0 *
                              fe_values.JxW(q_index));
            }
        }

      cell->get_dof_indices (local_dof_indices);
      constraints.distribute_local_to_global (cell_matrix,
                                              cell_rhs,
                                              local_dof_indices,
                                              system_matrix,
                                              system_rhs);
    }
}




template <int dim>
void Step6<dim>::solve ()
{
  SolverControl      solver_control (1000, 1e-12);
  SolverCG<>         solver (solver_control);

  PreconditionSSOR<> preconditioner;
  preconditioner.initialize(system_matrix, 1.2);

  solver.solve (system_matrix, solution, system_rhs,
                preconditioner);

  constraints.distribute (solution);
}



template <int dim>
void Step6<dim>::refine_grid ()
{
  estimate_kelly.reinit (triangulation.n_active_cells());
  KellyErrorEstimator<dim>::estimate (dof_handler,
                                      QGauss<dim-1>(fe.degree+2),
                                      typename FunctionMap<dim>::type(),
                                      solution,
                                      estimate_kelly);


  {
    estimates.block(0).reinit(triangulation.n_active_cells());
    estimates.collect_sizes();
    unsigned int i=0;
    for (typename Triangulation<dim>::active_cell_iterator cell = triangulation.begin_active();
         cell != triangulation.end(); ++cell,++i)
      cell->set_user_index(i);


    MeshWorker::IntegrationInfoBox<dim> info_box;
    const unsigned int n_gauss_points = dof_handler.get_fe().tensor_degree()+1;
    info_box.initialize_gauss_quadrature(n_gauss_points, n_gauss_points+1, n_gauss_points);

    AnyData solution_data;
    solution_data.add<const Vector<double>*>(&solution, "solution");

    info_box.cell_selector.add("solution", false, false, true);
    info_box.boundary_selector.add("solution", true, true, false);
    info_box.face_selector.add("solution", true, true, false);

    info_box.add_update_flags_boundary(update_quadrature_points);
    static MappingQ1<dim> mapping;
    info_box.initialize(fe, mapping , solution_data, solution);

    MeshWorker::DoFInfo<dim> dof_info(dof_handler);

    MeshWorker::Assembler::CellsAndFaces<double> assembler;
    AnyData out_data;
    out_data.add<BlockVector<double>*>(&estimates, "cells");
    assembler.initialize(out_data, false);

    Estimator<dim> integrator;

    MeshWorker::LoopControl ctrl;
    ctrl.faces_to_ghost = MeshWorker::LoopControl::both;

    MeshWorker::integration_loop<dim, dim> (dof_handler.begin_active(),
                                            dof_handler.end(),
                                            dof_info, info_box,
                                            integrator, assembler, ctrl);
  }


  {
    Functions::LSingularityFunction exact_solution;

    Vector<float> difference_per_cell (triangulation.n_active_cells());
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       exact_solution,
                                       difference_per_cell,
                                       QGauss<dim>(fe.degree+2),
                                       VectorTools::L2_norm);
    const double L2_error = VectorTools::compute_global_error(triangulation,
                                                              difference_per_cell,
                                                              VectorTools::L2_norm);
    VectorTools::integrate_difference (dof_handler,
                                       solution,
                                       exact_solution,
                                       difference_per_cell,
                                       QGauss<dim>(fe.degree+2),
                                       VectorTools::H1_norm);

    const double H1_error = VectorTools::compute_global_error(triangulation,
                                                              difference_per_cell,
                                                              VectorTools::H1_norm);
    const double estimate = estimate_kelly.l2_norm();
    std::cout << "      "
              << solution.size() << " "
              << L2_error << " "
              << H1_error << " "
              << estimate << " "
              << estimates.l2_norm()
              << std::endl;

  }


  if (adaptive)
    {

      GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                       estimates.block(0),
                                                       0.3, 0.0);
    }
  else
    {
      triangulation.set_all_refine_flags();
    }
}



template <int dim>
void Step6<dim>::output_results (const unsigned int cycle) const
{
  std::string filename = "solution-" + Utilities::to_string(cycle, 2) + ".vtu";
  std::ofstream output (filename.c_str());

  DataOut<dim> data_out;
  data_out.attach_dof_handler (dof_handler);

  Functions::LSingularityFunction exact_solution;
  Vector<double>  interpolated_exactsolution;
  interpolated_exactsolution.reinit(dof_handler.n_dofs());
  VectorTools::interpolate(dof_handler, exact_solution, interpolated_exactsolution);
  data_out.add_data_vector (interpolated_exactsolution, "exact_solution");

  data_out.add_data_vector (solution, "solution");
  data_out.add_data_vector (estimate_kelly, "kelly");
  data_out.add_data_vector (estimates, "estimator");
  data_out.build_patches ();

  data_out.write_vtu (output);
}



template <int dim>
void Step6<dim>::run ()
{
  for (unsigned int cycle=0; cycle<(adaptive? 12:6); ++cycle)
    {
      std::cout << " # Cycle: " << cycle << std::endl;
      if (cycle == 0)
        {
          GridGenerator::hyper_L (triangulation);
          triangulation.refine_global (1);
        }
      else
        triangulation.execute_coarsening_and_refinement ();

      setup_system ();
      assemble_system ();
      solve ();
      refine_grid ();
      output_results (cycle);
    }


}






int main ()
{

  try
    {
      Step6<2> laplace_problem_2d(2, false);
      std::cout << " # dofs Error_l2 Error_H1 Kelly_est RBE" << std::endl;
      laplace_problem_2d.run ();
    }
  catch (std::exception &exc)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Exception on processing: " << std::endl
                << exc.what() << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;

      return 1;
    }
  catch (...)
    {
      std::cerr << std::endl << std::endl
                << "----------------------------------------------------"
                << std::endl;
      std::cerr << "Unknown exception!" << std::endl
                << "Aborting!" << std::endl
                << "----------------------------------------------------"
                << std::endl;
      return 1;
    }

  return 0;
}
