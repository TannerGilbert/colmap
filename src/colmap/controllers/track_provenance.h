#pragma once

#include "colmap/controllers/matcher_cache.h"
#include "colmap/controllers/pairing.h"

#include <filesystem>
#include <functional>
#include <memory>

namespace colmap {

bool TrackProvenanceEnabled(
    const SequentialPairingOptions& options);

// Derive track provenance from the verified sequential-matcher database state.
// Direct consecutive image pairs and same/adjacent-frame rig pairs are kept as
// tracking/non-LC pairs. Other generated pairs are compared against transitive
// direct-track evidence; transitive rows stay non-LC and remaining candidate
// rows become LC.
void DeriveTrackProvenance(
    const std::shared_ptr<FeatureMatcherCache>& cache,
    const SequentialPairingOptions& options,
    const std::function<bool()>& is_stopped = nullptr);

void DeriveTrackProvenance(
    const std::filesystem::path& database_path,
    const SequentialPairingOptions& options,
    const std::function<bool()>& is_stopped = nullptr);

}  // namespace colmap
