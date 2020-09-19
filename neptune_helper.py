import neptune

APITOK = 'eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiMDNiMGRhYWMtNWY5OC00NTY1LWFiYTItOTdmZWNhMzQ2MThiIn0='

class NeptuneHelper():
    def __init__(self, exp_name = 'classification', params = None):
        neptune.init(project_qualified_name='martynawck/hackzurich2020', api_token=APITOK)
        neptune.create_experiment(name=exp_name, params=params)
        self.metrics = {}
        self.last_x = 0

    def log_metric(self, metric_name, y,  **kwargs):
        if 'x' not in kwargs:
            self.last_x += 1
            x = self.last_x
        else:
            x = kwargs['x']
        neptune.log_metric(metric_name, x, y)
        print(f'{x} | {metric_name} = {y}')