from datetime import datetime

class Metric:
    """
    A utility class to add metrics and convert the data into the expected MetricData format
    The timestamp associated with the metric starts after you create the object and you can time
    """

    def __init__(self, name):
        self.metric_name = name
        self.timestamp = datetime.now()

    def add_value(self, value, unit, count):
        self.value = value
        self.unit = unit
        self.count = count

    def add_count(self, value, count):
        self.value = value
        self.unit = 'Count'
        self.count = count

    def add_time_as_seconds(self, count):
        self.value = (datetime.now() - self.timestamp).total_seconds()
        self.unit = 'Seconds'
        self.count = count

    def add_time_as_milliseconds(self, count):
        self.value = (datetime.now() - self.timestamp).microseconds/1000
        self.unit = 'Milliseconds'
        self.count = count

    def add_time_as_microseconds(self, count):
        self.value = (datetime.now() - self.timestamp).microseconds
        self.unit = 'Microseconds'
        self.count = count

    def add_dimensions(self, dimensions):
        self.dimensions = dimensions

    def get_cw_metric_object(self):
        cw_metric_datum = dict()
        cw_metric_datum['MetricName'] = self.metric_name
        cw_metric_datum['Dimensions'] = self.dimensions
        ## We are going to aggregate metrics per minute
        cw_metric_datum['Timestamp'] = self.timestamp.replace(second=0,microsecond=0)
        cw_metric_datum['Value'] = self.value
        cw_metric_datum['Unit'] = self.unit
        cw_metric_datum['Count'] = self.count
        return cw_metric_datum

class MetricsFactory:
    """
    Metrics factory to supply metrics objects with configured dimenstions
    """

    def __init__(self, dimensions):
        self.dimension = dimensions

    def get_metric_object(self, name):
        m = Metric(name)
        m.add_dimensions(self.dimension)
        return m
