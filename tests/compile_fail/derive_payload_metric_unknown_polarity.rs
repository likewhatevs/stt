// `#[metric(polarity = Nonsense)]` — an unrecognized polarity ident
// must fail to compile. The macro accepts only the four `Polarity`
// variants (`HigherBetter`, `LowerBetter`, `Unknown`, and
// `TargetValue(<float>)`); anything else is rejected with a pointer
// to the expected variants so typos surface at compile time.
use ktstr::Payload;

#[derive(Payload)]
#[payload(binary = "metric_bad_polarity_bin")]
#[metric(name = "iops", polarity = Nonsense)]
#[allow(dead_code)]
struct MetricBadPolarityPayload;

fn main() {}
