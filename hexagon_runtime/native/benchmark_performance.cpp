// SPDX-License-Identifier: BSD-3-Clause
// Explicit benchmark policy. Normal inference leaves power votes to the caller.
#include <HAP_power.h>

namespace {
unsigned client;
bool active;

__attribute__((destructor)) void release() {
  if (active)
    HAP_power_destroy(&client);
}
} // namespace

// Called by the serialized resident session before its timed forward interval.
// A maximum vote is a request; thermal/platform limits can still lower clocks.
extern "C" bool pytorch_hexagon_request_benchmark_performance() {
  if (active)
    return true;
  HAP_power_request_t request{};
  request.type = HAP_power_set_apptype;
  request.apptype = HAP_POWER_COMPUTE_CLIENT_CLASS;
  if (HAP_power_set(&client, &request))
    return false;
  request = {};
  request.type = HAP_power_set_DCVS_v3;
  request.dcvs_v3.set_dcvs_enable = TRUE;
  request.dcvs_v3.dcvs_enable = FALSE;
  request.dcvs_v3.set_bus_params = TRUE;
  request.dcvs_v3.bus_params.min_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.bus_params.max_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.bus_params.target_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.set_core_params = TRUE;
  request.dcvs_v3.core_params.min_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.core_params.max_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.core_params.target_corner = HAP_DCVS_VCORNER_MAX;
  request.dcvs_v3.set_sleep_disable = TRUE;
  request.dcvs_v3.sleep_disable = TRUE;
  if (HAP_power_set(&client, &request)) {
    HAP_power_destroy(&client);
    client = 0;
    return false;
  }
  active = true;
  return true;
}
