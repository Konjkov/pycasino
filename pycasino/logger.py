import logging

logging.basicConfig(
    level=logging.INFO,
    filename='pycasino.log',
    filemode='w',
    format='%(message)s'
)

disclamer = """
 ------------------------------------------------------------------------------

                   #####
                  ##   ##    ##     ####   ##  #   ##   ####
                  ##        ####   ##      ##  ##  ##  ##  ##
                  ##       ##  ##   ####   ##  ### ##  ##  ##
                  ##       ######      ##  ##  ## ###  ##  ##
                  ##   ##  ##  ##  ##  ##  ##  ##  ##  ##  ##
                   #####   ##  ##   ####   ##  ##   #   ####

                              Python Quantum Package
                       v0.0.0 [Konjkov Vladimir] (30 August 2020)

    Main Authors : V.V. Konjkov

 ------------------------------------------------------------------------------
"""


class StreamToLogger:
    """Fake file-like stream object that redirects writes to a logger instance.
    https://stackoverflow.com/questions/19425736/how-to-redirect-stdout-and-stderr-to-logger-in-python
    """
    def __init__(self, logger, level):
        self.logger = logger
        self.level = level
        self.linebuf = ''

    def write(self, buf):
        for line in buf.rstrip().splitlines():
            self.logger.log(self.level, line.rstrip())

    def flush(self):
        pass
