#include "granite/TrajectorySet.hpp"

void granite::TrajectorySet::add_(const std::vector<my::math::Vec3> & data_point)
{
    if (_space != Space::Space3D)
        MY_USER_ERROR("Invalid function was called: add(std::vector<Vec3>).");

    _trajectories3d.push_back(std::move(data_point));
}

void granite::TrajectorySet::add_(const std::vector<my::math::Vec2> & data_point)
{
    if (_space != Space::Space2D)
        MY_USER_ERROR("Invalid function was called: add(std::vector<Vec2>).");

    _trajectories2d.push_back(std::move(data_point));
}

void granite::TrajectorySet::add(pybind11::array_t<float> data_point)
{
    pybind11::buffer_info buffer{data_point.request()};

    if (buffer.shape[0] != static_cast<pybind11::ssize_t>(_num_trajectories) ||
        buffer.shape.size() != 2)
        MY_USER_ERROR("Invalid trajectory data point was given to TrajectorySet instance.");

    if (buffer.shape[1] == 2)
    {
        push<my::math::Vec2>(_trajectories2d, buffer.ptr);
    }
    else if (buffer.shape[1] == 3)
    {
        push<my::math::Vec3>(_trajectories3d, buffer.ptr);
    }
    else
    {
        MY_USER_ERROR("Invalid trajectory data point was given to TrajectorySet instance.");
    }
}

granite::TrajectorySet::TrajectorySet(Space space, size_t num_trajectories)
    : _space{space}, _num_trajectories{num_trajectories}
{}

granite::TrajectorySet::TrajectorySet(pybind11::array_t<float> points)
{
    pybind11::buffer_info buffer{points.request()};

    if (buffer.shape.size() == 2)
    {
        _num_trajectories = static_cast<size_t>(buffer.shape[0]);

        if (buffer.shape[1] == 2)
            _space = Space::Space2D;
        else if (buffer.shape[1] == 3)
            _space = Space::Space3D;
        else
            MY_USER_ERROR("TrajectorySet does not support the given type of coordinates.");

        if (_space == Space::Space2D)
            push<my::math::Vec2>(_trajectories2d, buffer.ptr);
        else
            push<my::math::Vec3>(_trajectories3d, buffer.ptr);
    }
    else
        MY_USER_ERROR("Currently only 1 dimensional initial pointsets are supported.");
}

template <typename VEC_T>
void granite::TrajectorySet::push(std::vector<std::vector<VEC_T>> & target, void * bufferptr) const
{
    std::vector<VEC_T> dp(_num_trajectories);
    VEC_T * data{reinterpret_cast<VEC_T *>(bufferptr)};
    for (size_t i = 0; i < _num_trajectories; ++i) dp[i] = data[i];
    target.push_back(std::move(dp));
}

pybind11::array_t<float> granite::TrajectorySet::trajectory(size_t particle_id) const
{
    if (_space == Space::Space2D) return trajectory_(_trajectories2d, particle_id);
    return trajectory_(_trajectories3d, particle_id);
}

pybind11::array_t<float> granite::TrajectorySet::cloud() const
{
    if (_space == Space::Space2D) return cloud_(_trajectories2d);
    return cloud_(_trajectories3d);
}

pybind11::array_t<float> granite::TrajectorySet::totalCurvatures() const
{
    pybind11::array_t<float> result = pybind11::array_t<float>(_total_curvature.size());
    pybind11::buffer_info result_buffer{result.request()};
    float * data{reinterpret_cast<float *>(result_buffer.ptr)};
    for (size_t i{0}; i < _total_curvature.size(); ++i) data[i] = _total_curvature[i];
    return result;
}

pybind11::array_t<float> granite::TrajectorySet::individualLengths() const
{
    pybind11::array_t<float> result = pybind11::array_t<float>(_individual_lengths.size());
    pybind11::buffer_info result_buffer{result.request()};
    float * data{reinterpret_cast<float *>(result_buffer.ptr)};
    for (size_t i{0}; i < _individual_lengths.size(); ++i) data[i] = _individual_lengths[i];
    return result;
}

template <typename VEC_T>
pybind11::array_t<float>
granite::TrajectorySet::numpy_(pybind11::array::ShapeContainer container,
                               const std::vector<std::vector<VEC_T>> & trajectories) const
{
    const constexpr size_t DIM = sizeof(VEC_T) / sizeof(float);

    // create python numpy array and resize it
    pybind11::array_t<float> result =
        pybind11::array_t<float>(trajectories.size() * _num_trajectories * DIM);
    result.resize(container);

    // get data pointer
    pybind11::buffer_info result_buffer{result.request()};
    VEC_T * data{reinterpret_cast<VEC_T *>(result_buffer.ptr)};

    for (size_t i = 0; i < trajectories.size(); ++i)
        for (size_t j = 0; j < _num_trajectories; ++j)
            data[j * trajectories.size() + i] = trajectories[i][j];

    return result;
}

template <typename VEC_T>
pybind11::array_t<float>
granite::TrajectorySet::cloud_(const std::vector<std::vector<VEC_T>> & trajectories) const
{
    return numpy_(
        {
            static_cast<pybind11::ssize_t>(trajectories.size() * _num_trajectories), //
            static_cast<pybind11::ssize_t>(sizeof(VEC_T) / sizeof(float))            //
        },
        trajectories);
}

pybind11::array_t<float> granite::TrajectorySet::trajectories() const
{
    if (_space == Space::Space2D) return trajectories_(_trajectories2d);
    return trajectories_(_trajectories3d);
}

template <typename VEC_T>
pybind11::array_t<float>
granite::TrajectorySet::trajectories_(const std::vector<std::vector<VEC_T>> & trajectories) const
{
    return numpy_(
        {
            static_cast<pybind11::ssize_t>(_num_trajectories),            //
            static_cast<pybind11::ssize_t>(trajectories.size()),          //
            static_cast<pybind11::ssize_t>(sizeof(VEC_T) / sizeof(float)) //
        },
        trajectories);
}

template <typename VEC_T>
pybind11::array_t<float>
granite::TrajectorySet::trajectory_(const std::vector<std::vector<VEC_T>> & trajectories,
                                    size_t particle_id) const
{
    const constexpr size_t DIM = sizeof(VEC_T) / sizeof(float);

    // create python numpy array and resize it
    pybind11::array_t<float> result{pybind11::array_t<float>(DIM * trajectories.size())};
    result.resize({
        static_cast<pybind11::ssize_t>(trajectories.size()), //
        static_cast<pybind11::ssize_t>(DIM)                  //
    });

    // get data pointer
    pybind11::buffer_info result_buffer{result.request()};
    VEC_T * data{reinterpret_cast<VEC_T *>(result_buffer.ptr)};

    // copy data over
    for (size_t i = 0; i < trajectories.size(); ++i) data[i] = trajectories[i][particle_id];

    return result;
}

pybind11::array_t<float> granite::TrajectorySet::volumeInfo(std::string volume_key) const
{
    if (auto volume = _volume_data.find(volume_key); volume != _volume_data.end())
    {
        const std::vector<std::vector<float>> & vdata{(*volume).second};

        // create python numpy array and resize it
        pybind11::array_t<float> result =
            pybind11::array_t<float>(vdata.size() * _num_trajectories);
        result.resize({static_cast<pybind11::ssize_t>(_num_trajectories), //
                       static_cast<pybind11::ssize_t>(vdata.size())});

        // get data pointer
        pybind11::buffer_info result_buffer{result.request()};
        float * data{reinterpret_cast<float *>(result_buffer.ptr)};

        for (size_t i = 0; i < vdata.size(); ++i)
            for (size_t j = 0; j < _num_trajectories; ++j) data[j * vdata.size() + i] = vdata[i][j];

        return result;
    }
    else
        MY_USER_ERROR("TrajectorySet: Volume with given key '" << volume_key
                                                               << "' does not exist!");
}

pybind11::array_t<float> granite::TrajectorySet::computeInfo(size_t compute_id) const
{
    if (compute_id < _num_additional_compute)
    {
        const std::vector<std::vector<float>> & vdata{_compute_data[compute_id]};

        // create python numpy array and resize it
        pybind11::array_t<float> result =
            pybind11::array_t<float>(vdata.size() * _num_trajectories);
        result.resize({static_cast<pybind11::ssize_t>(_num_trajectories), //
                       static_cast<pybind11::ssize_t>(vdata.size())});

        // get data pointer
        pybind11::buffer_info result_buffer{result.request()};
        float * data{reinterpret_cast<float *>(result_buffer.ptr)};

        for (size_t i = 0; i < vdata.size(); ++i)
            for (size_t j = 0; j < _num_trajectories; ++j) data[j * vdata.size() + i] = vdata[i][j];

        return result;
    }
    else
        MY_USER_ERROR("TrajectorySet: Volume with given id does not exist!");
}

pybind11::array_t<float> granite::TrajectorySet::curvatures() const
{
    const std::vector<std::vector<float>> & vdata{_curvature};

    // create python numpy array and resize it
    pybind11::array_t<float> result = pybind11::array_t<float>(vdata.size() * _num_trajectories);
    result.resize({static_cast<pybind11::ssize_t>(_num_trajectories), //
                   static_cast<pybind11::ssize_t>(vdata.size())});

    // get data pointer
    pybind11::buffer_info result_buffer{result.request()};
    float * data{reinterpret_cast<float *>(result_buffer.ptr)};

    for (size_t i = 0; i < vdata.size(); ++i)
        for (size_t j = 0; j < _num_trajectories; ++j) data[j * vdata.size() + i] = vdata[i][j];

    return result;
}