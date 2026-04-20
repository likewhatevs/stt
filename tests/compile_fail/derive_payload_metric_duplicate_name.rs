// Two `#[metric(name = "iops", ...)]` declarations with the same
// name must fail to compile. The runtime `resolve_polarities`
// pipeline uses last-wins semantics when it builds a name→hint map,
// so a duplicate silently shadows the first hint instead of
// surfacing the user's likely copy-paste typo. The derive rejects
// duplicates up-front so the error lands in the attribute that
// created the collision.
use ktstr::Payload;

#[derive(Payload)]
#[payload(binary = "metric_dup_bin")]
#[metric(name = "iops", polarity = HigherBetter)]
#[metric(name = "iops", polarity = LowerBetter)]
#[allow(dead_code)]
struct MetricDupPayload;

fn main() {}
