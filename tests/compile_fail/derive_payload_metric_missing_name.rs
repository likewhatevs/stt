// `#[metric(...)]` without a `name = "..."` key must fail to compile.
// The parser requires every MetricHint to carry a name so that
// runtime metric extraction can key its polarity + unit lookup by
// that name.
use ktstr::Payload;

#[derive(Payload)]
#[payload(binary = "metric_no_name_bin")]
#[metric(polarity = HigherBetter)]
#[allow(dead_code)]
struct MetricNoNamePayload;

fn main() {}
