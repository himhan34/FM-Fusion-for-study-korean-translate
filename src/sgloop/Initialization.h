#ifndef INITIALIZATION_H
#define INITIALIZATION_H

#include <mapping/SemanticMapping.h>
#include <sgloop/Graph.h>


namespace fmfusion
{

    bool init_scene_graph(const Config &config, const std::string &scene_dir,
                          std::shared_ptr<Graph> &src_graph)
    {
        std::vector<InstanceId> instance_idxs;
        std::vector<InstancePtr> instances;

        auto src_map = std::make_shared<SemanticMapping>(
            SemanticMapping(config.mapping_cfg, config.instance_cfg));
        src_graph = std::make_shared<Graph>(config.graph);

        // Prepare map
        src_map->load(scene_dir);
        src_map->extract_bounding_boxes();
        src_map->export_instances(instance_idxs, instances);
        assert(instance_idxs.size()>0 && instances.size()>0);

        // Prepare graph
        src_graph->initialize(instances);
        src_graph->construct_edges();
        src_graph->construct_triplets();
        // fmfusion::DataDict src_data_dict = src_graph->extract_data_dict();

        return true;
    }

}

#endif // INITIALIZATION_H
