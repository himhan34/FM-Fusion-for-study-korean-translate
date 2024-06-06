#include <memory>

#include "Common.h"
#include "back_end/reglib.h"
// #include "tools/g3reg_api.h"
#include "sgloop/Graph.h"


namespace Registration
{

    void pruneInsOutliers(const fmfusion::RegistrationConfig &config,
                        const std::vector<fmfusion::NodePtr> &src_nodes,
                        const std::vector<fmfusion::NodePtr> &ref_nodes,
                        const std::vector<fmfusion::NodePair> &match_pairs,
                        std::vector<bool> &pruned_true_masks);
    
    
} // namespace Registration

