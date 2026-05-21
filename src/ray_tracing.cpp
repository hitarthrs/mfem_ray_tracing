#include "ray_tracing.hpp"

#include <cmath>
#include <vector>

namespace
{

void BuildTSamples(double t_entry, double t_exit, double dt, std::vector<double> &t_samples)
{
    t_samples.clear();

    if (dt <= 0.0)  // bad step -> auto spacing
    {
        dt = (t_exit - t_entry) * 0.01;
        if (dt <= 0.0)
        {
            dt = 1e-12;
        }
    }

    t_samples.push_back(t_entry);  // first sample at mesh entry

    double t = t_entry;
    while (t + dt < t_exit)  // march forward by dt
    {
        t += dt;
        t_samples.push_back(t);
    }

    constexpr double tol = 1e-12;
    if (t_samples.back() < t_exit - tol)  // ensure we hit t_exit
    {
        t_samples.push_back(t_exit);
    }
}

bool FindCellIndicesAt(const Ray &ray,
                       mfem::Mesh &mesh,
                       const std::vector<double> &t_samples,
                       std::vector<int> &elem_ids)
{
    const int n = static_cast<int>(t_samples.size());
    if (n == 0)
    {
        elem_ids.clear();
        return false;
    }

    const int sd = mesh.SpaceDimension();
    mfem::DenseMatrix point_mat(sd, n);  // sd rows, one column per t sample

    for (int j = 0; j < n; ++j)  // column j = physical point at t_samples[j]
    {
        mfem::Vector point;
        ray.Evaluate(t_samples[j], point);
        for (int d = 0; d < sd; ++d)
        {
            point_mat(d, j) = point[d];
        }
    }

    mfem::Array<int> elem_arr(n);
    mfem::Array<mfem::IntegrationPoint> ip(n);
    mesh.FindPoints(point_mat, elem_arr, ip);  // one batched lookup for all samples

    elem_ids.resize(n);
    for (int j = 0; j < n; ++j)  // copy out (-1 if point not in mesh)
    {
        elem_ids[j] = elem_arr[j];
    }
    return true;
}

// Turn sample arrays into crossing segments whenever the cell index changes.
void BuildCrossingsFromSamples(const std::vector<double> &t_samples,
                             const std::vector<int> &elem_ids,
                             std::vector<RayCellCrossings> &crossings)
{
    crossings.clear();
    if (t_samples.empty() || elem_ids.empty())
    {
        return;
    }

    if (elem_ids[0] < 0)
    {
        return;
    }

    int seg_start = 0;

    for (std::size_t j = 1; j < elem_ids.size(); ++j)
    {
        // If the cell index is -1, the point is not inside any cell and the ray has left the mesh.
        if (elem_ids[j] < 0)
        {
            RayCellCrossings seg;
            seg.cell_index = elem_ids[seg_start];
            seg.t_entry = t_samples[seg_start];
            seg.t_exit = t_samples[j];
            crossings.push_back(seg);
            return;
        }

        // If the cell index changes, add a crossing segment from the last valid cell to the current point.
        if (elem_ids[j] != elem_ids[j - 1])
        {
            RayCellCrossings seg;
            seg.cell_index = elem_ids[j - 1];
            seg.t_entry = t_samples[seg_start];
            seg.t_exit = t_samples[j];
            crossings.push_back(seg);
            seg_start = static_cast<int>(j);
        }
    }

    if (elem_ids[seg_start] >= 0)
    {
        RayCellCrossings seg;
        seg.cell_index = elem_ids[seg_start];
        seg.t_entry = t_samples[seg_start];
        seg.t_exit = t_samples.back();
        crossings.push_back(seg);
    }
}

}  // namespace

bool TraceFindPoints(const Ray &ray,
                     mfem::Mesh &mesh,
                     double dt,
                     std::vector<RayCellCrossings> &crossings)
{
    double t_entry = 0.0;
    double t_exit = 0.0;
    if (!IntersectAABB(ray, mesh, t_entry, t_exit))
    {
        crossings.clear();
        return false;
    }

    std::vector<double> t_samples;
    BuildTSamples(t_entry, t_exit, dt, t_samples);

    std::vector<int> elem_ids;
    if (!FindCellIndicesAt(ray, mesh, t_samples, elem_ids))
    {
        crossings.clear();
        return false;
    }

    BuildCrossingsFromSamples(t_samples, elem_ids, crossings);
    return !crossings.empty();
}
