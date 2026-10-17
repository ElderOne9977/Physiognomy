import pathlib, sys, traceback
out = pathlib.Path(__file__).parent / "experiments" / "semantic_calibration_phase314" / "_test_run_output.txt"
out.parent.mkdir(parents=True, exist_ok=True)
err_path = pathlib.Path(__file__).parent / "experiments" / "semantic_calibration_phase314" / "_test_run_err.txt"
orig_out, orig_err = sys.stdout, sys.stderr
buf_out = open(out, "w", encoding="utf-8")
buf_err = open(err_path, "w", encoding="utf-8")
sys.stdout = buf_out
sys.stderr = buf_err
exit_code = 0
try:
    import test_nested_semantic_calibration_phase314 as T
    T.main()
except SystemExit as e:
    exit_code = int(e.code) if e.code is not None else 0
    print(f"SystemExit {exit_code}")
    if exit_code != 0:
        pass
except Exception:
    traceback.print_exc()
    exit_code = 2
finally:
    buf_out.close()
    buf_err.close()
    sys.stdout = orig_out
    sys.stderr = orig_err
    pathlib.Path(__file__).parent.joinpath("experiments/semantic_calibration_phase314/_test_run_exit.txt").write_text(str(exit_code), encoding="utf-8")
    print(f"capture done exit={exit_code} -> {out}")