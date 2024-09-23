
#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>

#include "open3d/utility/Logging.h"
#include "open3d/geometry/PointCloud.h"
#include "open3d/pipelines/integration/UniformTSDFVolume.h"
#include "open3d/pipelines/integration/ScalableTSDFVolume.h"
#include "open3d/pipelines/integration/MarchingCubesConst.h"

namespace fmfusion
{
    // class UniformTSDFVolume;
    typedef open3d::pipelines::integration::ScalableTSDFVolume ScalableTSDFVolume;
    typedef std::shared_ptr<open3d::geometry::PointCloud> PointCloudPtr;
    typedef open3d::pipelines::integration::TSDFVolumeColorType TSDFVolumeColorType;

    class SubVolume : public ScalableTSDFVolume {
    public:
        SubVolume(double voxel_length,
                        double sdf_trunc,
                        TSDFVolumeColorType color_type,
                        int volume_unit_resolution = 16,
                        int depth_sampling_stride = 4);
        ~SubVolume() override;

    public:
        // std::shared_ptr<geometry::PointCloud> ExtractWeightedPointCloud(const float min_weight=0.0);

        /// @brief Generate scan cloud from depth image. 
        ///         And query observed voxel points from the volume.
        bool query_observed_points(const PointCloudPtr &cloud_scan,
                                PointCloudPtr &cloud_observed,
                                const float max_dist=0.98f);

        //todo: inaccurate
        /// @brief  Get the centroid of all volume units origin.
        /// @return 
        Eigen::Vector3d get_centroid();

    protected:
        Eigen::Vector3i LocateVolumeUnit(const Eigen::Vector3d &point) {
            return Eigen::Vector3i((int)std::floor(point(0) / volume_unit_length_),
                                (int)std::floor(point(1) / volume_unit_length_),
                                (int)std::floor(point(2) / volume_unit_length_));
        };

        Eigen::Vector3d GetNormalAt(const Eigen::Vector3d &p);

        double GetTSDFAt(const Eigen::Vector3d &p);

    };


}

