import json


manifest_json_string = '''
 {
     "nodeGraph" : {
       "envelopeVersion" : "2021-01-01",
       "packages" : [
           {
               "name" : "panorama::abstract_rtsp_media_source",
               "version" : "1.0"
           }
       ],
       "nodes" : [
           {
               "name" : "rtsp_camera_node", 
               "interface" : "panorama::abstract_rtsp_media_source.rtsp_v1_interface",
               "overridable" : true,
               "overrideMandatory" : false,
               "launch" : "onAppStart"
           }
       ],
       "edges" : [
       ]
   }
 }
'''

override_json_string = '''
{
    "nodeGraphOverrides" : {
        "envelopeVersion" : "2021-01-01",
        "packages" : [
        ],
        "nodes" : [
        ],
        "nodeOverrides" : [
            {
                "replace" : "rtsp_camera_node",
                "with" : [
                ]
            }
        ]
    }
}
'''


class App:

    def __init__(self, account_id, name):
        self.account_id = account_id
        self.name = name
        self.manifest = json.loads(manifest_json_string)
        self.override = dict()

    def get_name(self):
        return self.name

    def get_manifest(self):
        return self.manifest

    def get_manifest_as_str(self):
        return json.dumps(self.manifest)

    def generate_override(self, data_sources):
        def get_replace_with_node(data_source_name):
            return {"name": data_source_name}

        def get_node(data_source_name):
            return {
                "name": data_source_name,
                "interface": self.account_id + "::" + data_source_name + "." + data_source_name,
                "overridable": True,
                "overrideMandatory": False,
                "launch": "onAppStart"
            }

        def get_package(data_source_name):
            return {
                "name": self.account_id + "::" + data_source_name,
                "version": "1.0"
            }

        self.override = json.loads(override_json_string)
        for data_source in data_sources:
            self.override['nodeGraphOverrides']['packages'].append(get_package(data_source))
            self.override['nodeGraphOverrides']['nodes'].append(get_node(data_source))
            self.override['nodeGraphOverrides']['nodeOverrides'][0]['with'].append(get_replace_with_node(data_source))

    def get_override(self):
        return self.override

    def get_override_as_str(self):
        return json.dumps(self.override)