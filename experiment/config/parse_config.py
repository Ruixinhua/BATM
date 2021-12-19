import copy
import logging
from collections import defaultdict
from pathlib import Path
from experiment.config import Configuration, arch_default_config
from logger import setup_logging


class ConfigParser:
    def __init__(self, config: Configuration, modification: dict = None):
        """
        class to parse configuration json file. Handles hyper-parameters for training, initializations of modules,
        checkpoint saving and logging module.
        :param config: Dict containing configurations, hyper-parameters for training. Normal saved in configs directory.
        :param modification: Dict keychain:value, specifying position values to be replaced from config dict.
        Timestamp is being used as default
        """
        # load config file and apply modification.
        self.config = config
        if modification:
            self.config.update(modification)

        # set save_dir where training model and log will be saved.
        save_dir = Path(self.config.save_dir)
        run_name = self.config.run_name

        self._save_dir = save_dir / "models" / run_name
        self._log_dir = save_dir / "log" / run_name
        # configure logging module
        self.log_levels = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}
        # make directory for saving checkpoints and log
        self._save_dir.mkdir(parents=True, exist_ok=True)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        setup_logging(self.log_dir)
        # save updated config file to the checkpoint directory
        self.config.save_config(self._save_dir)

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @classmethod
    def from_args(cls, args, options: list = None):
        """
        Initialize this class from some cli arguments. Used in train, test.
        """
        for opt in options:
            default = opt.default if hasattr(opt, "default") else None
            args.add_argument(*opt.flags, default=default, type=opt.type)
        if not isinstance(args, tuple):
            args = args.parse_args()
        # parse custom cli options into dictionary
        modification = defaultdict()
        if hasattr(args, "arch_type") and args.arch_type is not None:
            modification["arch_config"] = arch_default_config(args.arch_type)  # setup default arch params
        for opt in options:
            name = opt.flags[-1].replace("--", "")  # acquire param name
            if opt.target:
                if opt.target not in modification:
                    modification[opt.target] = {}
                if getattr(args, name):
                    modification[opt.target][name] = getattr(args, name)  # setup custom params values
            else:
                if getattr(args, name):
                    modification[name] = getattr(args, name)  # setup custom params values
        if hasattr(args, "resume") and args.resume is not None:
            config_file = Path(args.resume).parent / "config.json"
            config = Configuration.from_json_file(config_file)
        else:
            config = Configuration(**modification)
        return cls(config)

    def init_obj(self, module_config: str, module: object, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('trainer_config', module, a, b=1)`
        is equivalent to
        `object = module.module_name(a, b=1)`
        """
        module_args = copy.deepcopy(getattr(self.config, module_config))
        module_args.update(kwargs)
        module_name = module_args.pop("type")
        return getattr(module, module_name)(*args, **module_args)

    def get_logger(self, name, verbosity=2):
        msg_verbosity = f"verbosity option{verbosity} is invalid. Valid options are {self.log_levels.keys()}."
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return getattr(self.config, name)
