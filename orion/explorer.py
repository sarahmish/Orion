import json
import logging
import math
import pickle
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import similaritymeasures as sm
from bson import ObjectId
from bson.errors import InvalidId
from gridfs import GridFS
from mlblocks import MLPipeline
from mongoengine import connect
from pip._internal.operations import freeze
from pymongo.database import Database
from scipy import stats

from orion import model
from orion.data import load_signal

LOGGER = logging.getLogger(__name__)


class OrionExplorer:

    def __init__(self, database='orion', **kwargs):
        self.database = database
        self._db = connect(database, **kwargs)
        self._software_versions = list(freeze.freeze())
        self._fs = GridFS(Database(self._db, self.database))

    def drop_database(self):
        self._db.drop_database(self.database)

    def _list(self, model, exclude_=None, **kwargs):
        query = {
            key: value
            for key, value in kwargs.items()
            if value is not None
        }
        documents = model.find(exclude_=exclude_, **query)

        data = pd.DataFrame([
            document.to_mongo()
            for document in documents
        ]).rename(columns={'_id': model.__name__.lower() + '_id'})

        for column in exclude_ or []:
            if column in data:
                del data[column]

        return data

    def add_dataset(self, name, entity_id=None):
        return model.Dataset.find_or_insert(
            name=name,
            entity_id=entity_id
        )

    def get_datasets(self, name=None, entity_id=None):
        return self._list(
            model.Dataset,
            name=name,
            entity_id=entity_id
        )

    def get_dataset(self, dataset):
        try:
            _id = ObjectId(dataset)
            return model.Dataset.last(id=_id)
        except InvalidId:
            found_dataset = model.Dataset.last(name=dataset)
            if found_dataset is None:
                raise ValueError('Dataset not found: {}'.format(dataset))
            else:
                return found_dataset

    def add_signal(self, name, dataset, start_time=None, stop_time=None, location=None,
                   timestamp_column=None, value_column=None, user_id=None):

        location = location or name
        data = load_signal(location, None, timestamp_column, value_column)
        timestamps = data['timestamp']
        if not start_time:
            start_time = timestamps.min()

        if not stop_time:
            stop_time = timestamps.max()

        dataset = self.get_dataset(dataset)

        return model.Signal.find_or_insert(
            name=name,
            dataset=dataset,
            start_time=start_time,
            stop_time=stop_time,
            data_location=location,
            timestamp_column=timestamp_column,
            value_column=value_column,
            created_by=user_id
        )

    def get_signals(self, name=None, dataset=None):
        return self._list(
            model.Signal,
            name=name,
            dataset=dataset
        )

    def load_signal(self, signal):
        path_or_name = signal.data_location or signal.name
        LOGGER.info("Loading dataset %s", path_or_name)
        data = load_signal(path_or_name, None, signal.timestamp_column, signal.value_column)
        if signal.start_time:
            data = data[data['timestamp'] >= signal.start_time].copy()

        if signal.stop_time:
            data = data[data['timestamp'] <= signal.stop_time].copy()

        return data

    def add_pipeline(self, name, path, user_id=None):
        with open(path, 'r') as pipeline_file:
            pipeline_json = json.load(pipeline_file)

        return model.Pipeline.find_or_insert(
            name=name,
            mlpipeline=pipeline_json,
            created_by=user_id
        )

    def get_pipelines(self, name=None):
        return self._list(
            model.Pipeline,
            dataset__name=name,
        )

    def get_pipeline(self, pipeline):
        try:
            _id = ObjectId(pipeline)
            return model.Pipeline.last(id=_id)
        except InvalidId:
            found_pipeline = model.Pipeline.last(name=pipeline)
            if found_pipeline is None:
                raise ValueError('Pipeline not found: {}'.format(pipeline))
            else:
                return found_pipeline

    def load_pipeline(self, pipeline):
        LOGGER.info("Loading pipeline %s", pipeline.name)
        return MLPipeline.from_dict(pipeline.mlpipeline)

    def get_experiments(self):
        return self._list(
            model.Experiment
        )

    def get_dataruns(self, experiment=None):
        return self._list(
            model.Datarun,
            exclude_=['software_versions'],
            experiment=experiment
        )

    def start_datarun(self, experiment, signal):
        return model.Datarun.insert(
            experiment=experiment,
            signal=signal,
            start_time=datetime.utcnow(),
            software_versions=self._software_versions,
            status='running'
        )

    def end_datarun(self, datarun, events, status='done'):
        try:
            for i in range(len(events)):
                if len(events[i])==3:
                    model.Event.insert(
                        datarun=datarun,
                        start_time=int(events[i][0]),
                        stop_time=int(events[i][1]),
                        score=events[i][2]
                    )
                else:
                    event = model.Event.insert(
                        datarun=datarun,
                        start_time=int(events[i][0]),
                        stop_time=int(events[i][1]),
                        score=events[i][2],
                        tag=events[i][3]
                    )
                    model.Comment.insert(
                        event=event,
                        text='similarity matching'
                    )
        except Exception:
            LOGGER.exception('Error storing datarun %s events', datarun.id)
            status = 'error'

        datarun.end_time = datetime.utcnow()
        datarun.status = status
        datarun.events = len(events)
        datarun.save()

    def add_comment(self, event, text, user_id):
        model.Comment.insert(
            event=event,
            text=text,
            created_by=user_id,
        )

    def get_events(self, datarun=None):
        events = self._list(
            model.Event,
            datarun=datarun
        )

        if events.empty:
            return events

        comments = list()
        for event in events.event_id:
            events_count = model.Comment.objects(event=event).count()
            comments.append(events_count)

        events['comments'] = comments

        return events

    def get_comments(self, datarun=None, event=None):
        if event is None:
            query = {'datarun': datarun}
        else:
            query = {'id': event}

        events = self._list(
            model.Event,
            exclude_=['insert_time'],
            **query
        )

        if events.empty:
            return pd.DataFrame()

        comments = self._list(
            model.Comment,
            event__in=list(events.event_id)
        )
        comments = comments.rename(columns={'event': 'event_id'})

        return events.merge(comments, how='inner', on='event_id')

    def find_pattern_in_sequence(self, pattern, sequence):
        step_size_var = 60
        step_size = max(1, len(pattern) // step_size_var)
        i = 0
        similarity_dtw = list()
        pattern_data = np.zeros((len(pattern), 2))
        pattern_data[:, 0] = np.arange(len(pattern))
        pattern_data[:, 1] = pattern[:]
        while i < len(sequence) - len(pattern):
            sequence_data = np.zeros((len(pattern), 2))
            sequence_data[:, 0] = np.arange(len(pattern))
            sequence_data[:, 1] = sequence[i:i + len(pattern)]
            dtw, _ = sm.dtw(pattern_data, sequence_data)
            similarity_dtw = similarity_dtw + [dtw / len(pattern)]
            i += step_size

        s = stats.zscore(similarity_dtw)
        below = pd.Series(s < -4)
        index_below = np.argwhere(below)

        for idx in index_below.flatten():
            below[max(0, idx - max(1, len(pattern)//step_size)):min(idx + max(1, len(pattern)//step_size) + 1, len(below))] = True

        shift = below.shift(1).fillna(False)
        change = below != shift
        index = below.index

        starts = (index[below & change]).tolist()
        ends = (index[~below & change]).tolist()
        severity = [min(similarity_dtw[start:end]) for start, end in np.array([starts, ends]).T]

        for i in range(len(starts)):
            min_dist_index = similarity_dtw[starts[i]:ends[i]].index(min(similarity_dtw[starts[i]:ends[i]])) + starts[i]
            starts[i] = min_dist_index * step_size
            ends[i] = starts[i] + len(pattern)

        if len(ends) == len(starts) - 1:
            ends.append(len(sequence) - 1)

        return np.array([starts, ends, severity]).T

    def _merge_sequences(self, sequences, signal):

        sorted_sequences = sorted(sequences, key=lambda entry: entry[0])
        new_sequences = [[signal] + sorted_sequences[0]]

        for sequence in sorted_sequences[1:]:
            prev_sequence = new_sequences[-1]
            if int(sequence[0]) <= int(prev_sequence[2]):
                new_sequences[-1] = (prev_sequence[0], prev_sequence[1], max(prev_sequence[2], sequence[1]), prev_sequence[3])
            else:
                new_sequences.append([signal] + sequence)
        return new_sequences

    def get_sequences(self, sig):
        tag_list = ['Problem', 'Normal', 'Previously seen', 'Investigate', 'Do not investigate',
                    'Postpone']
        dataruns_signal = model.Datarun.find(signal=sig)
        events_problem = self._list(model.Event,
                                    datarun__in=[datarun for datarun in dataruns_signal],
                                    tag__in=tag_list)
        if len(events_problem) > 0:
            events_problem = np.array(events_problem[['start_time', 'stop_time', 'tag']])
            events_problem = self._merge_sequences(events_problem.tolist(), sig)
        return events_problem

    def run_experiment(self, project, pipeline, dataset, user_id=None, similarity_check=False):
        project = project
        pipeline = self.get_pipeline(pipeline)
        dataset = self.get_dataset(dataset)

        experiment = model.Experiment.find_or_insert(
            project=project,
            pipeline=pipeline,
            dataset=dataset,
            created_by=user_id
        )

        mlpipeline = self.load_pipeline(pipeline)
        signals = model.Signal.find(dataset=dataset.id)

        events_sim = []

        if similarity_check == 'all':
            for sig in signals:
                events_problem = self.get_sequences(sig)
                if len(events_problem) > 0:
                    events_sim.extend(events_problem)

        elif isinstance(similarity_check, list):
            if len(similarity_check) > 0:
                for sig in similarity_check:
                    signal = model.Signal.find(name=sig)[0]
                    events_problem = self.get_sequences(signal)
                    if len(events_problem) > 0:
                        events_sim.extend(events_problem)

        for signal in signals:
            if similarity_check == 'same':
                events_sim = []
                events_problem = self.get_sequences(signal)
                if len(events_problem) > 0:
                    events_sim.extend(events_problem)

            events_sim = np.array(events_sim)

            try:
                data = self.load_signal(signal)
                datarun = self.start_datarun(experiment, signal)

                LOGGER.info("Fitting the pipeline")
                mlpipeline.fit(data)

                outputs = ["default"]

                try:
                    output_names = mlpipeline.get_output_names('visualization')
                    outputs.append('visualization')
                except ValueError:
                    output_names = []

                LOGGER.info("Finding events")
                pipeline_output = mlpipeline.predict(data, output_=outputs)

                if not isinstance(pipeline_output, tuple):
                    events = pipeline_output
                else:
                    events = pipeline_output[0]
                    if output_names:
                        # There might be multiple `default` outputs before the `visualization`
                        # outputs in the pipeline_output tuple, thus we get the last entries
                        # corresponding to visualization
                        visualization = pipeline_output[-len(output_names):]

                        visualization_dict = dict(zip(output_names, visualization))
                        for name, value in visualization_dict.items():
                            kwargs = {
                                "filename": '{}-{}.pkl'.format(datarun.id, name),
                                "datarun_id": datarun.id,
                                "variable": name
                            }
                            with self._fs.new_file(**kwargs) as f:
                                pickle.dump(value, f)

                events = events.tolist()

                for anomaly in events_sim:
                    data_pattern = self.load_signal(anomaly[0])
                    pattern = data_pattern.loc[
                        (data_pattern['timestamp'] >= int(anomaly[1])) & (data_pattern['timestamp'] <= int(anomaly[2])), [
                            'value']]
                    pattern = list(np.array(pattern).flatten())
                    sequence = list(np.array(data['value']).flatten())
                    similar_sequences = self.find_pattern_in_sequence(pattern, sequence)

                    for similar_sequence in similar_sequences:
                        for event in events:
                            if max(0, min(data.iloc[int(similar_sequence[1])]['timestamp'], event[1]) - max(data.iloc[int(similar_sequence[0])]['timestamp'], event[0]) + 1) > 0:
                                if len(event)==3 and event[2] < similar_sequence[2]:
                                    continue
                                else:
                                    events.remove(event)
                                    events.append([data['timestamp'][int(similar_sequence[0])], data['timestamp'][int(similar_sequence[1])], 0, anomaly[3], similar_sequence[2]])

                status = 'done'

            except Exception:
                LOGGER.exception('Error running datarun %s', datarun.id)
                events = list()
                status = 'error'

            self.end_datarun(datarun, events, status)

            LOGGER.info("%s events found in %s", len(events),
                        datarun.end_time - datarun.start_time)

        return experiment
