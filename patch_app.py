from pathlib import Path
import shutil

APP_PATH = Path(r"D:\art_studio_app\app.py")
BAK_PATH = Path(r"D:\art_studio_app\app.py.bak")

SCHEDULER_TAB_CODE = '''

# ---------------------------------------------------------------------------
# 5. Tab 2: Instagram Feed Scheduler & Batch Interleaver
# ---------------------------------------------------------------------------
def interleave_groups(
    group_a: list[tuple[str, Image.Image]],
    group_b: list[tuple[str, Image.Image]],
    count_a: int,
    count_b: int,
) -> list[tuple[str, str, Image.Image]]:
    """Interleave group_a and group_b in cycles of (count_a, count_b) until both are exhausted."""
    result: list[tuple[str, str, Image.Image]] = []
    idx_a, idx_b = 0, 0
    len_a, len_b = len(group_a), len(group_b)
    while idx_a < len_a or idx_b < len_b:
        for _ in range(max(0, count_a)):
            if idx_a < len_a:
                lbl, im = group_a[idx_a]
                result.append(("Group A", lbl, im))
                idx_a += 1
        for _ in range(max(0, count_b)):
            if idx_b < len_b:
                lbl, im = group_b[idx_b]
                result.append(("Group B", lbl, im))
                idx_b += 1
    return result


def is_low_sat_sketch(pil_img: Image.Image) -> bool:
    rgb = np.array(pil_img.convert("RGB"))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return float(np.mean(hsv[:, :, 1])) < 28.0 and float(np.mean(hsv[:, :, 1] > 65)) < 0.08


with tab_scheduler:
    st.subheader("Feed Scheduler & Batch Interleaver")
    st.caption(
        "Mix Pastel and Pencil artworks in a repeatable ratio (e.g., 2 Pastel : 2 Pencil or 2 Pencil : 1 Pastel), "
        "preview the 3-column grid order, and export numbered files (`01_...`, `02_...`) for one-shot uploading."
    )

    sched_source = st.radio(
        "Where should the scheduler load images from?",
        ["Upload Two Groups (Group A & Group B)", "Auto-Split Local `output/` Folder (Color vs. Pencil/Sketch)"],
        key="sched_source_radio",
    )

    group_a_items: list[tuple[str, Image.Image]] = []
    group_b_items: list[tuple[str, Image.Image]] = []

    if sched_source.startswith("Upload Two Groups"):
        up_c1, up_c2 = st.columns(2, gap="large")
        with up_c1:
            files_a = st.file_uploader(
                "Group A (e.g., Pastel / Color artworks)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="sched_up_a",
            )
            if files_a:
                for f in sorted(files_a, key=lambda x: x.name):
                    group_a_items.append((f.name, Image.open(BytesIO(f.getvalue())).convert("RGB")))
        with up_c2:
            files_b = st.file_uploader(
                "Group B (e.g., Pencil / Sketch artworks)",
                type=["png", "jpg", "jpeg", "webp"],
                accept_multiple_files=True,
                key="sched_up_b",
            )
            if files_b:
                for f in sorted(files_b, key=lambda x: x.name):
                    group_b_items.append((f.name, Image.open(BytesIO(f.getvalue())).convert("RGB")))
    else:
        out_files = sorted(
            [p for p in OUTPUT_DIR.glob("*") if p.is_file() and p.suffix.lower() in SUPPORTED_IMAGE_EXTS]
        )
        for p in out_files:
            im = Image.open(BytesIO(p.read_bytes())).convert("RGB")
            if is_low_sat_sketch(im):
                group_b_items.append((p.name, im))
            else:
                group_a_items.append((p.name, im))
        st.info(
            f"Detected in `output/`: **{len(group_a_items)}** Color/Pastel (Group A) and "
            f"**{len(group_b_items)}** Pencil/Sketch (Group B)."
        )

    pat_c1, pat_c2, pat_c3, pat_c4 = st.columns([1.4, 1.4, 1.4, 1.6])
    with pat_c1:
        pat_preset = st.selectbox(
            "Quick pattern",
            ["2 Group A : 2 Group B", "2 Group B : 1 Group A", "1 Group A : 1 Group B", "Custom Counts"],
            index=0,
            key="sched_pat_preset",
        )
    if pat_preset == "2 Group A : 2 Group B":
        def_a, def_b = 2, 2
    elif pat_preset == "2 Group B : 1 Group A":
        def_a, def_b = 1, 2
    elif pat_preset == "1 Group A : 1 Group B":
        def_a, def_b = 1, 1
    else:
        def_a, def_b = 2, 1

    with pat_c2:
        take_a = int(
            st.number_input(
                "Group A per cycle",
                min_value=0,
                max_value=20,
                value=def_a,
                step=1,
                disabled=(pat_preset != "Custom Counts"),
                key="sched_take_a",
            )
        )
    with pat_c3:
        take_b = int(
            st.number_input(
                "Group B per cycle",
                min_value=0,
                max_value=20,
                value=def_b,
                step=1,
                disabled=(pat_preset != "Custom Counts"),
                key="sched_take_b",
            )
        )
    with pat_c4:
        sched_prefix = st.text_input("Output filename prefix", value="post", key="sched_prefix_in")

    if not group_a_items and not group_b_items:
        st.caption("Add images to Group A and/or Group B above to preview and export your scheduled sequence.")
    elif take_a == 0 and take_b == 0:
        st.warning("Set at least one group count per cycle above 0.")
    else:
        mixed = interleave_groups(group_a_items, group_b_items, take_a, take_b)
        st.success(f"Scheduled **{len(mixed)}** artworks ({len(group_a_items)} Group A + {len(group_b_items)} Group B).")

        sched_zip_buf = BytesIO()
        pad_width = max(2, len(str(len(mixed))))
        clean_sprefix = sanitize_stem(sched_prefix, "post")

        with zipfile.ZipFile(sched_zip_buf, "w", compression=zipfile.ZIP_DEFLATED) as szf:
            for idx_1, (grp_tag, orig_lbl, im) in enumerate(mixed, start=1):
                seq_str = str(idx_1).zfill(pad_width)
                orig_stem = sanitize_stem(orig_lbl, f"item_{seq_str}")
                out_name = f"{seq_str}_{clean_sprefix}_{orig_stem}.jpg"
                enc_bytes, _, _ = encode_output_image(im, "JPG", 95)
                szf.writestr(out_name, enc_bytes)

        dl_c1, dl_c2 = st.columns([2.0, 2.0])
        with dl_c1:
            st.download_button(
                label=f"Download Scheduled Sequence ({len(mixed)} files .zip)",
                data=sched_zip_buf.getvalue(),
                file_name=f"scheduled_{clean_sprefix}_{len(mixed)}_items.zip",
                mime="application/zip",
                type="primary",
                key="sched_download_zip_btn",
            )
        with dl_c2:
            if st.button("Also Save Numbered Sequence to `output/scheduled/`", key="sched_save_disk_btn"):
                sched_dir = OUTPUT_DIR / "scheduled"
                sched_dir.mkdir(parents=True, exist_ok=True)
                for idx_1, (_, orig_lbl, im) in enumerate(mixed, start=1):
                    seq_str = str(idx_1).zfill(pad_width)
                    orig_stem = sanitize_stem(orig_lbl, f"item_{seq_str}")
                    out_name = f"{seq_str}_{clean_sprefix}_{orig_stem}.jpg"
                    enc_bytes, _, _ = encode_output_image(im, "JPG", 95)
                    (sched_dir / out_name).write_bytes(enc_bytes)
                st.success(f"Saved {len(mixed)} numbered files to `{sched_dir}`.")

        st.markdown("#### 3-Column Feed Order Preview (Top-to-Bottom Posting Sequence)")
        for r_start in range(0, len(mixed), 3):
            row_slice = mixed[r_start : r_start + 3]
            g_cols = st.columns(3, gap="medium")
            for c_i, (grp_tag, orig_lbl, im) in enumerate(row_slice):
                pos_num = r_start + c_i + 1
                with g_cols[c_i]:
                    st.image(
                        proportional_preview(im, 220),
                        caption=f"#{str(pos_num).zfill(pad_width)} [{grp_tag}] {orig_lbl}",
                    )
'''


def main() -> None:
    if not APP_PATH.exists():
        raise FileNotFoundError(f"Missing {APP_PATH}")

    shutil.copy2(APP_PATH, BAK_PATH)
    src = APP_PATH.read_text(encoding="utf-8")

    if "tab_scheduler" in src:
        print("Tab 2 (Feed Scheduler) is already present in app.py.")
        return

    anchor = 'top_c1, top_c2, top_c3, top_c4 = st.columns([1.2, 1.6, 1.2, 1.6])'
    pos = src.find(anchor)
    if pos == -1:
        raise RuntimeError("Could not find main UI anchor in app.py.")

    # Indent existing main area under `with tab_editor:`
    head = src[:pos]
    body = src[pos:]
    indented_body = "\n".join(("    " + line if line.strip() else line) for line in body.splitlines())

    tabs_header = (
        'tab_editor, tab_scheduler = st.tabs(["1. Batch Edit & Enhance", "2. Feed Scheduler & Mixer"])\n\n'
        'with tab_editor:\n'
    )

    patched = head + tabs_header + indented_body + "\n" + SCHEDULER_TAB_CODE
    APP_PATH.write_text(patched, encoding="utf-8")
    print(f"Added Tab 2 (Feed Scheduler & Mixer) to {APP_PATH} (backup at {BAK_PATH}).")


if __name__ == "__main__":
    main()