from tqdm.auto import tqdm
import sys
import logging


def make_tqdm_pbar(data_loader, desc: str=""):
    pbar = tqdm(
        data_loader,
        desc=desc,
        total=len(data_loader),
        dynamic_ncols=True,
        leave=False,
        disable=False,
    )
    return pbar


class TqdmLoggingHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            msg = self.format(record)
            tqdm.write(msg, file=self.stream)
            self.flush()
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception:
            self.handleError(record)