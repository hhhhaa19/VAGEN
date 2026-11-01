import importlib, multiprocessing, sys, types

multiprocessing.cpu_count = lambda: 8

def import_with_stubs(name):
    while True:
        try:
            return importlib.import_module(name)
        except ModuleNotFoundError as e:
            missing = e.name
            if missing.startswith("vagen"):
                raise
            print(f"[stub] {missing}", file=sys.stderr)
            parts = missing.split(".")
            for i in range(1, len(parts) + 1):
                sys.modules.setdefault(".".join(parts[:i]), types.ModuleType(".".join(parts[:i])))
            mod = sys.modules[missing]
            def _stub_attr(attr, _m=missing):
                if attr.startswith("__"):
                    raise AttributeError(attr)
                return type(attr, (), {})
            mod.__getattr__ = _stub_attr
            for k in list(sys.modules):
                if k.startswith("vagen"):
                    del sys.modules[k]

SokobanEnvConfig = import_with_stubs("vagen.env.sokoban.env_config").SokobanEnvConfig

if __name__ == "__main__":
    import contextlib, io
    config = SokobanEnvConfig()
    mode = sys.argv[1] if len(sys.argv) > 1 else "snippet"
    if mode == "snippet":
        print(config.config_id())
        with contextlib.redirect_stdout(io.StringIO()):
            seeds = config.generate_seeds(10)
        print(seeds)
    else:
        with contextlib.redirect_stdout(io.StringIO()):
            train = config.generate_seeds(10000)
            test = config.generate_seeds(128)
        print("train size", len(train), "test size", len(test))
        print("test seeds first 20:", test[:20])
        print("test == train[:128]:", test == train[:128])
        print("test seeds also in train:", len(set(test) & set(train)), "/", len(test))
