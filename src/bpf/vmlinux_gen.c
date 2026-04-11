/*
 * Generate vmlinux.h from a BTF source file.
 *
 * Replaces the bpftool dependency by using libbpf's btf_dump API
 * directly. Called from build.rs via FFI.
 *
 * Usage: generate_vmlinux_h(btf_path, output_path)
 *   btf_path:    path to vmlinux ELF or /sys/kernel/btf/vmlinux
 *   output_path: path to write vmlinux.h
 *   returns:     0 on success, -1 on error
 */

#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <bpf/btf.h>

static void btf_dump_printf(void *ctx, const char *fmt, va_list args)
{
	vfprintf((FILE *)ctx, fmt, args);
}

int generate_vmlinux_h(const char *btf_path, const char *output_path)
{
	struct btf *btf;
	struct btf_dump *dump;
	FILE *out;
	int err = 0;
	__u32 i, n;

	btf = btf__parse(btf_path, NULL);
	err = libbpf_get_error(btf);
	if (err) {
		fprintf(stderr, "vmlinux_gen: btf__parse(%s): %s\n",
			btf_path, strerror(-err));
		return -1;
	}

	out = fopen(output_path, "w");
	if (!out) {
		fprintf(stderr, "vmlinux_gen: fopen(%s): %s\n",
			output_path, strerror(errno));
		btf__free(btf);
		return -1;
	}

	fprintf(out, "#ifndef __VMLINUX_H__\n");
	fprintf(out, "#define __VMLINUX_H__\n\n");
	fprintf(out, "#ifndef BPF_NO_PRESERVE_ACCESS_INDEX\n");
	fprintf(out, "#pragma clang attribute push(__attribute__((preserve_access_index)), apply_to = record)\n");
	fprintf(out, "#endif\n\n");

	dump = btf_dump__new(btf, btf_dump_printf, out, NULL);
	err = libbpf_get_error(dump);
	if (err) {
		fprintf(stderr, "vmlinux_gen: btf_dump__new: %s\n",
			strerror(-err));
		fclose(out);
		btf__free(btf);
		return -1;
	}

	n = btf__type_cnt(btf);
	for (i = 1; i < n; i++) {
		err = btf_dump__dump_type(dump, i);
		if (err) {
			fprintf(stderr, "vmlinux_gen: dump type %u: %s\n",
				i, strerror(-err));
			break;
		}
	}

	fprintf(out, "\n#ifndef BPF_NO_PRESERVE_ACCESS_INDEX\n");
	fprintf(out, "#pragma clang attribute pop\n");
	fprintf(out, "#endif\n\n");
	fprintf(out, "#endif /* __VMLINUX_H__ */\n");

	btf_dump__free(dump);
	fclose(out);
	btf__free(btf);
	return err ? -1 : 0;
}
