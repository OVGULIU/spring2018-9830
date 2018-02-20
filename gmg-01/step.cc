/* ---------------------------------------------------------------------
 *
 * Copyright (C) 2000 - 2016 by the deal.II authors
 *
 * This file is part of the deal.II library.
 *
 * The deal.II library is free software; you can use it, redistribute
 * it, and/or modify it under the terms of the GNU Lesser General
 * Public License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 * The full text of the license can be found in the file LICENSE at
 * the top level of the deal.II distribution.
 *
 * ---------------------------------------------------------------------

 *
 * Author: Conrad Clevenger
 */



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



#include <deal.II/multigrid/mg_constrained_dofs.h>
#include <deal.II/multigrid/multigrid.h>
#include <deal.II/multigrid/mg_transfer.h>
#include <deal.II/multigrid/mg_tools.h>
#include <deal.II/multigrid/mg_coarse.h>
#include <deal.II/multigrid/mg_smoother.h>
#include <deal.II/multigrid/mg_matrix.h>



#include <fstream>
#include <iostream>

#include <deal.II/fe/fe_q.h>
#include <deal.II/grid/grid_out.h>


#include <deal.II/lac/constraint_matrix.h>

#include <deal.II/grid/grid_refinement.h>

#include <deal.II/numerics/error_estimator.h>

using namespace dealii;



template <int dim>
class MultigridStep6
{
public:
  MultigridStep6 (const unsigned int deg);
  ~MultigridStep6 ();

  void run ();

private:
  void setup_system ();
  void assemble_system ();
  void assemble_multigrid ();
  void solve ();
  void refine_grid ();
  void output_results (const unsigned int cycle) const;

  Triangulation<dim>           triangulation;
  const SphericalManifold<dim> boundary;

  FE_Q<dim>            fe;
  unsigned int         degree;
  DoFHandler<dim>      mg_dof_handler;

  ConstraintMatrix     constraints;

  SparsityPattern      sparsity_pattern;
  SparseMatrix<double> system_matrix;

  Vector<double>       solution;
  Vector<double>       system_rhs;

  // GMG specific
  MGLevelObject<SparsityPattern>       mg_sparsity_patterns;
  MGLevelObject<SparsityPattern>       mg_interface_sparsity_patterns;
  MGLevelObject<SparseMatrix<double> > mg_matrices;
  MGLevelObject<SparseMatrix<double> > mg_interface_matrices;
  MGConstrainedDoFs                    mg_constrained_dofs;
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
MultigridStep6<dim>::MultigridStep6 (const unsigned int deg)
  :
    triangulation(Triangulation<dim>::limit_level_difference_at_vertices),
    fe (deg),
    degree(deg),
    mg_dof_handler (triangulation)
{}



template <int dim>
MultigridStep6<dim>::~MultigridStep6 ()
{
  triangulation.clear ();
}



template <int dim>
void MultigridStep6<dim>::setup_system ()
{
  mg_dof_handler.distribute_dofs (fe);
  mg_dof_handler.distribute_mg_dofs ();

  solution.reinit (mg_dof_handler.n_dofs());
  system_rhs.reinit (mg_dof_handler.n_dofs());


  constraints.clear ();
  DoFTools::make_hanging_node_constraints (mg_dof_handler,
                                           constraints);


  VectorTools::interpolate_boundary_values (mg_dof_handler,
                                            0,
                                            Functions::ZeroFunction<dim>(),
                                            constraints);


  constraints.close ();

  DynamicSparsityPattern dsp(mg_dof_handler.n_dofs());
  DoFTools::make_sparsity_pattern(mg_dof_handler,
                                  dsp,
                                  constraints,
                                  /*keep_constrained_dofs = */ false);

  sparsity_pattern.copy_from(dsp);

  system_matrix.reinit (sparsity_pattern);


  // GMG Specific
  mg_constrained_dofs.clear();
  mg_constrained_dofs.initialize(mg_dof_handler);

  std::set<types::boundary_id>  dirichlet_boundary_ids;
  dirichlet_boundary_ids.insert(0);
  mg_constrained_dofs.make_zero_boundary_constraints(mg_dof_handler, dirichlet_boundary_ids);

  const unsigned int n_levels = triangulation.n_global_levels();

  mg_matrices.resize(0, n_levels-1);
  mg_matrices.clear_elements ();
  mg_interface_matrices.resize(0, n_levels-1);
  mg_interface_matrices.clear_elements ();
  mg_sparsity_patterns.resize(0, n_levels-1);
  mg_interface_sparsity_patterns.resize(0, n_levels-1);

  for (unsigned int level=0; level<n_levels; ++level)
  {
    {
      DynamicSparsityPattern dsp(mg_dof_handler.n_dofs(level),
                                 mg_dof_handler.n_dofs(level));
      MGTools::make_sparsity_pattern(mg_dof_handler, dsp, level);
      mg_sparsity_patterns[level].copy_from (dsp);
      mg_matrices[level].reinit(mg_sparsity_patterns[level]);
    }
    {
      DynamicSparsityPattern dsp(mg_dof_handler.n_dofs(level),
                                 mg_dof_handler.n_dofs(level));
      MGTools::make_interface_sparsity_pattern(mg_dof_handler, mg_constrained_dofs, dsp, level);
      mg_interface_sparsity_patterns[level].copy_from(dsp);
      mg_interface_matrices[level].reinit(mg_interface_sparsity_patterns[level]);
    }
  }
}



template <int dim>
void MultigridStep6<dim>::assemble_system ()
{
  const QGauss<dim>  quadrature_formula(degree+1);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  typename DoFHandler<dim>::active_cell_iterator
      cell = mg_dof_handler.begin_active(),
      endc = mg_dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit (cell);

    for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
    {
      const double current_coefficient = coefficient<dim>
          (fe_values.quadrature_point (q_index));
      for (unsigned int i=0; i<dofs_per_cell; ++i)
      {
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          cell_matrix(i,j) += (current_coefficient *
                               fe_values.shape_grad(i,q_index) *
                               fe_values.shape_grad(j,q_index) *
                               fe_values.JxW(q_index));

        cell_rhs(i) += (fe_values.shape_value(i,q_index) *
                        1.0 *
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

// Assemble System on each multigrid level
template <int dim>
void MultigridStep6<dim>::assemble_multigrid ()
{
  // Set up temporary constraint matrices for each level that describe
  // both the boundary and refinement edge indices
  std::vector<ConstraintMatrix> boundary_constraints (triangulation.n_global_levels());
  for (unsigned int level=0; level<triangulation.n_global_levels(); ++level)
  {
    IndexSet dofset;
    DoFTools::extract_locally_relevant_level_dofs (mg_dof_handler, level, dofset);
    boundary_constraints[level].reinit(dofset);
    boundary_constraints[level].add_lines (mg_constrained_dofs.get_refinement_edge_indices(level));
    boundary_constraints[level].add_lines (mg_constrained_dofs.get_boundary_indices(level));
    boundary_constraints[level].close ();
  }

  // The following is very similar to assemble_system()
  const QGauss<dim>  quadrature_formula(degree+1);

  FEValues<dim> fe_values (fe, quadrature_formula,
                           update_values    |  update_gradients |
                           update_quadrature_points  |  update_JxW_values);

  const unsigned int   dofs_per_cell = fe.dofs_per_cell;
  const unsigned int   n_q_points    = quadrature_formula.size();

  FullMatrix<double>   cell_matrix (dofs_per_cell, dofs_per_cell);
  Vector<double>       cell_rhs (dofs_per_cell);

  std::vector<types::global_dof_index> local_dof_indices (dofs_per_cell);

  // iterate over all cells, not just active ones
  typename DoFHandler<dim>::cell_iterator
      cell = mg_dof_handler.begin(),
      endc = mg_dof_handler.end();
  for (; cell!=endc; ++cell)
  {
    const unsigned int level = cell->level();
    cell_matrix = 0;
    cell_rhs = 0;

    fe_values.reinit (cell);

    for (unsigned int q_index=0; q_index<n_q_points; ++q_index)
    {
      const double current_coefficient = coefficient<dim>
          (fe_values.quadrature_point (q_index));
      // No need for cell_rhs
      for (unsigned int i=0; i<dofs_per_cell; ++i)
        for (unsigned int j=0; j<dofs_per_cell; ++j)
          cell_matrix(i,j) += (current_coefficient *
                               fe_values.shape_grad(i,q_index) *
                               fe_values.shape_grad(j,q_index) *
                               fe_values.JxW(q_index));
    }

    cell->get_mg_dof_indices (local_dof_indices);
    boundary_constraints[level].distribute_local_to_global (cell_matrix,
                                                            local_dof_indices,
                                                            mg_matrices[level]);
    // Fill in interface matrices
    for (unsigned int i=0; i<dofs_per_cell; ++i)
      for (unsigned int j=0; j<dofs_per_cell; ++j)
        if (mg_constrained_dofs.is_interface_matrix_entry(level, local_dof_indices[i], local_dof_indices[j]))
          mg_interface_matrices[level].add(local_dof_indices[i],local_dof_indices[j],cell_matrix(i,j));
  }
}



template <int dim>
void MultigridStep6<dim>::solve ()
{
  SolverControl                     solver_control (1000, 1e-12);
  SolverCG<Vector<double> >         solver (solver_control);

  // Build the transfer matrices
  MGTransferPrebuilt<Vector<double> > mg_transfer(mg_constrained_dofs);
  mg_transfer.build_matrices(mg_dof_handler);

  // Create the coarse mesh solver
  SparseMatrix<double>  &coarse_matrix = mg_matrices[0];
  SolverControl coarse_solver_control (1000, 1e-10, false, false);
  SolverCG<Vector<double> > coarse_solver(coarse_solver_control);
  PreconditionIdentity id;
  MGCoarseGridIterativeSolver<Vector<double>, SolverCG<Vector<double> >, SparseMatrix<double> , PreconditionIdentity>
      coarse_grid_solver(coarse_solver, coarse_matrix, id);

  // Create the smoothers
  typedef PreconditionJacobi<SparseMatrix<double> > Smoother;
  MGSmootherPrecondition<SparseMatrix<double>, Smoother, Vector<double> > mg_smoother;
  mg_smoother.initialize(mg_matrices, Smoother::AdditionalData(0.66667));
  mg_smoother.set_steps(2);

  // Create the actual preconditioner
  mg::Matrix<Vector<double> > mg_matrix(mg_matrices);
  mg::Matrix<Vector<double> > mg_interface_in(mg_interface_matrices);
  mg::Matrix<Vector<double> > mg_interface_out(mg_interface_matrices);

  Multigrid<Vector<double> > mg(mg_matrix,
                                coarse_grid_solver,
                                mg_transfer,
                                mg_smoother,
                                mg_smoother);
  mg.set_edge_matrices(mg_interface_out, mg_interface_in);

  PreconditionMG<dim, Vector<double>, MGTransferPrebuilt<Vector<double> > >
      preconditioner(mg_dof_handler, mg, mg_transfer);

  solver.solve (system_matrix, solution, system_rhs,
                preconditioner);
  std::cout << "   CG converged in " << solver_control.last_step() << " iterations." << std::endl;

  constraints.distribute (solution);
}



template <int dim>
void MultigridStep6<dim>::refine_grid ()
{
  Vector<float> estimated_error_per_cell (triangulation.n_active_cells());

  KellyErrorEstimator<dim>::estimate (mg_dof_handler,
                                      QGauss<dim-1>(degree+1),
                                      typename FunctionMap<dim>::type(),
                                      solution,
                                      estimated_error_per_cell);

  GridRefinement::refine_and_coarsen_fixed_number (triangulation,
                                                   estimated_error_per_cell,
                                                   0.3, 0.03);

  triangulation.execute_coarsening_and_refinement ();
}



template <int dim>
void MultigridStep6<dim>::output_results (const unsigned int cycle) const
{
  DataOut<dim> data_out;
  data_out.attach_dof_handler (mg_dof_handler);
  data_out.add_data_vector (solution, "solution");
  data_out.build_patches ();
  {
    std::ostringstream filename;
    filename << "solution-"
             << cycle
             << ".vtk";
    std::ofstream output (filename.str().c_str());
    data_out.write_vtk (output);
  }

  GridOut grid_out;
  grid_out.write_mesh_per_processor_as_vtu(triangulation,
                                           "grid-"+Utilities::int_to_string(cycle),
                                           true);
}



template <int dim>
void MultigridStep6<dim>::run ()
{
  unsigned int n_cycles = 5;
  for (unsigned int cycle=0; cycle<n_cycles; ++cycle)
  {
    std::cout << "Cycle " << cycle << ':' << std::endl;

    if (cycle == 0)
    {
      GridGenerator::hyper_ball (triangulation);

      triangulation.set_all_manifold_ids_on_boundary(0);
      triangulation.set_manifold (0, boundary);

      triangulation.refine_global (1);
    }
    else
    {
      refine_grid ();
      //triangulation.refine_global();
    }


    std::cout << "   Number of active cells:       "
              << triangulation.n_active_cells()
              << std::endl;

    setup_system ();

    std::cout << "   Number of degrees of freedom: "
              << mg_dof_handler.n_dofs()
              << std::endl;

    assemble_system ();
    assemble_multigrid ();
    solve ();
    output_results (cycle);
  }
}



int main ()
{

  try
  {
    MultigridStep6<2> laplace_problem_2d(2);
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
