// `#[ktstr_test(config = "...")]` paired with the default scheduler
// (Payload::KERNEL_DEFAULT — no `config_file_def`) must fail at
// compile time. The macro emits a `const _: () = { ... };` block that
// const-evaluates `(scheduler).config_file_def().is_some()` against
// the macro-known `config_set` flag and panics on mismatch.
use ktstr::ktstr_test;

#[ktstr_test(config = "{}")]
fn config_without_def(_ctx: &ktstr::scenario::Ctx) -> anyhow::Result<ktstr::assert::AssertResult> {
    Ok(ktstr::assert::AssertResult::pass())
}

fn main() {}
