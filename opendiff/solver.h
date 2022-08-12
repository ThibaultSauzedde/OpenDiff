// void PetscVecToEigen(const Vec &pvec, unsigned int nrow, unsigned int ncol, Eigen::MatrixXd &emat)
// {
//     PetscScalar *pdata;
//     // Returns a pointer to a contiguous array containing this
//     processor's portion
//         // of the vector data. For standard vectors this doesn't use any copies.
//         // If the the petsc vector is not in a contiguous array then it will copy
//         // it to a contiguous array.
//         VecGetArray(pvec, &pdata);
//     // Make the Eigen type a map to the data. Need to be mindful of
//     anything that
//         // changes the underlying data location like re-allocations.
//         emat = Eigen::Map<Eigen::MatrixXd>(pdata, nrow, ncol);
//     VecRestoreArray(pvec, &pdata);
// }
