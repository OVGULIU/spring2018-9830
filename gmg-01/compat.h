namespace dealii
{
inline
bool
is_interface_matrix_entry (const MGConstrainedDoFs &mgcd,
			   const unsigned int level,
                                              const types::global_dof_index i,
                                              const types::global_dof_index j)
{
  const IndexSet &interface_dofs_on_level
    = mgcd.get_refinement_edge_indices(level);

  return interface_dofs_on_level.is_element(i)   // at_refinement_edge(i)
         &&
         !interface_dofs_on_level.is_element(j)  // !at_refinement_edge(j)
         &&
    !mgcd.is_boundary_index(level, i)      // !on_boundary(i)
         &&
    !mgcd.is_boundary_index(level, j);     // !on_boundary(j)
}


namespace MGTools
{
  template <typename DoFHandlerType, typename SparsityPatternType>
  void
  make_interface_sparsity_pattern (const DoFHandlerType    &dof,
                                   const MGConstrainedDoFs &mg_constrained_dofs,
                                   SparsityPatternType     &sparsity,
                                   const unsigned int      level)
  {
    const types::global_dof_index n_dofs = dof.n_dofs(level);
    (void)n_dofs;

    Assert (sparsity.n_rows() == n_dofs,
            ExcDimensionMismatch (sparsity.n_rows(), n_dofs));
    Assert (sparsity.n_cols() == n_dofs,
            ExcDimensionMismatch (sparsity.n_cols(), n_dofs));

    const unsigned int dofs_per_cell = dof.get_fe().dofs_per_cell;
    std::vector<types::global_dof_index> dofs_on_this_cell(dofs_per_cell);
    typename DoFHandlerType::cell_iterator cell = dof.begin(level),
                                           endc = dof.end(level);
    for (; cell!=endc; ++cell)
      if (cell->is_locally_owned_on_level())
        {
          cell->get_mg_dof_indices (dofs_on_this_cell);
          for (unsigned int i=0; i<dofs_per_cell; ++i)
            for (unsigned int j=0; j<dofs_per_cell; ++j)
              if (is_interface_matrix_entry(mg_constrained_dofs,level,dofs_on_this_cell[i],dofs_on_this_cell[j]))
                sparsity.add (dofs_on_this_cell[i],
                              dofs_on_this_cell[j]);
        }
  }
    }
}
