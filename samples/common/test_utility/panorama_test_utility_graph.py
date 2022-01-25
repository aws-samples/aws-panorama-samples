import sys
import os
import json
import re

# ---

re_pattern_package_fullname = r"([A-Za-z0-9_-]+)::([A-Za-z0-9_-]+)"
re_pattern_account_id = r"([0-9]+)"
re_pattern_stock_package_name = r"(abstract_rtsp_media_source|hdmi_data_sink)"
re_pattern_interface_fullname = r"([A-Za-z0-9_-]+)::([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)"
re_pattern_edge_fullname = r"([A-Za-z0-9_-]+)\.([A-Za-z0-9_-]+)"
re_pattern_edge_parameter_node_name = r"([A-Za-z0-9_-]+)"

# ---

def load_json_file(filepath):
    with open(filepath) as fd:
        return json.load(fd)

# ---

class PackageBase:
    pass

class JsonPackage(PackageBase):
    def __init__( self, filepath ):
        self.d = load_json_file(filepath)

class AbstractRtspMediaSourcePackage(PackageBase):
    def __init__(self):
        self.d = { 
            "nodePackage" : {
                "envelopeVersion": "2021-01-01",
                "name": "abstract_rtsp_media_source",
                "version": "1.0",
                "description": "",
                "assets" : [
                    # placeholder information for stock package
                    {
                        "name": "rtsp_v1_asset",
                        "implementations": [
                            {
                                "type": "system",
                                "assetUri":"source/video/camera/rtsp/source_rtsp"
                            }
                        ]
                    }
                ],
                "interfaces" : [
                    {
                        "name": "rtsp_v1_interface",
                        "category": "media_source",
                        "asset": "rtsp_v1_asset",
                        "outputs": [
                            {
                                "name": "video_out",
                                "type": "media",
                            },
                        ],
                    },
                ],
            }
        }

class HdmiDataSinkPackage(PackageBase):
    def __init__(self):
        self.d = { 
            "nodePackage" : {
                "envelopeVersion": "2021-01-01",
                "name": "hdmi_data_sink",
                "version": "1.0",
                "description": "",
                "assets" : [
                    # placeholder information for stock package
                    {
                        "name": "hdmi0_asset",
                        "implementations": [
                            {
                                "type": "data_sink",
                                "assetUri": "",
                                "descriptorUri": ""
                            }
                        ]
                    }
                ],
                "interfaces" : [
                    {
                        "name": "hdmi0",
                        "category": "data_sink",
                        "asset": "hdmi0_asset",
                        "inputs": [
                            {
                                "name": "video_in",
                                "type": "media",
                            },
                        ],
                    },
                ],
            }
        }

# ---

class Node:
    def __init__(self):
        pass

class PackagedNode(Node):

    def __init__( self, interface_elm, asset_elm ):
        Node.__init__(self)
        
        self.interface_elm = interface_elm
        self.asset_elm = asset_elm
    
    def lookup_input_output( self, list_name, name ):

        for elm in self.interface_elm[list_name]:
            if elm["name"] == name:
                return elm
        
        interface_name = self.interface_elm["name"]
        raise ValueError( f"'{name}' not found in interface '{interface_name}.{list_name}'" )

class BusinessLogicContainerNode(PackagedNode):

    def __init__( self, interface_elm, asset_elm ):
        
        PackagedNode.__init__( self, interface_elm, asset_elm )
        
        self.inputs = {}
        self.outputs = {}

    def connect_producer( self, input_name, producer_node, producer_output_name ):
    
        print( "Connecting producer", input_name, producer_node, producer_output_name )
        
        if isinstance( producer_node, PackagedNode ):
            
            input_elm = self.lookup_input_output( "inputs", input_name )
            output_elm = producer_node.lookup_input_output( "outputs", producer_output_name )
        
            input_type = input_elm["type"]
            output_type = output_elm["type"]
            if input_type != output_type:
                raise ValueError( f"Interface input/output types mismatch {input_type} != {output_type}" )

        self.inputs[input_name] = producer_node

    def connect_consumer( self, output_name, consumer_node, consumer_input_name ):

        if isinstance( consumer_node, PackagedNode ):

            output_elm = self.lookup_input_output( "outputs", output_name )
            input_elm = consumer_node.lookup_input_output( "inputs", consumer_input_name )
        
            input_type = input_elm["type"]
            output_type = output_elm["type"]
            if input_type != output_type:
                raise ValueError( f"Interface input/output types mismatch {input_type} != {output_type}" )

        self.outputs[output_name] = consumer_node

class ModelNode(PackagedNode):
    def __init__( self, interface_elm, asset_elm ):
        PackagedNode.__init__( self, interface_elm, asset_elm )

class MediaSourceRtspCameraNode(PackagedNode):
    def __init__( self, interface_elm, asset_elm ):
        PackagedNode.__init__( self, interface_elm, asset_elm )

class HdmiDataSinkNode(PackagedNode):
    def __init__( self, interface_elm, asset_elm ):
        PackagedNode.__init__( self, interface_elm, asset_elm )

class ParameterNode(Node):

    def __init__( self, node_elm ):

        t = node_elm["interface"]
        v = node_elm["value"]
        
        types = {
            "float32" : float,
            "int32" : int,
            "string" : str,
            "boolean" : bool,
        }
        
        if t not in types:
            raise ValueError( f"Unknown parameter type {t}" )
        
        if not isinstance( v, types[t] ):
            raise TypeError( f"Expected type is {t} but value is {type(v)}" )

        self.value = v

        self.node_elm = node_elm
    
    def lookup_input_output( self, list_name, name ):
        
        print( "self.node_elm", self.node_elm )
        
        for elm in self.interface_elm[list_name]:
            if elm["name"] == name:
                return elm
        
        interface_name = self.interface_elm["name"]
        raise ValueError( f"'{name}' not found in interface '{interface_name}.{list_name}'" )
        

# ---

class Graph:

    def __init__(self):
        self.packages = {}
        self.nodes = {}
        self.business_logic_node = None

    def load( self, app_dir_top, app_name ):

        self.app_dir_top = app_dir_top
        self.app_name = app_name

        graph_filepath = os.path.join( app_dir_top, "graphs", app_name, "graph.json" )

        print( "Loading graph:", graph_filepath )
        print( "" )

        graph_json = load_json_file(graph_filepath)

        print( "Loading packages" )

        # load dependent package JSON files, and descriptor JSON files
        for package_elm in graph_json["nodeGraph"]["packages"]:
            
            package_fullname = package_elm["name"]
            package_version = package_elm["version"]

            print( f"Processing {package_fullname}" )

            re_result = re.match( re_pattern_package_fullname, package_fullname )
            if re_result:
                account_id = re_result.group(1)
                package_name = re_result.group(2)

                if account_id == "panorama":
                    if package_name=="abstract_rtsp_media_source":
                        self.packages[package_name] = AbstractRtspMediaSourcePackage()
                    elif package_name=="hdmi_data_sink":
                        self.packages[package_name] = HdmiDataSinkPackage()
                    else:
                        raise ValueError( f"Unsupported stock package name : {package_name}" )
                else:
                    # FIXME : check if this matches actual account id.
                    self.load_package_from_json( account_id, package_name, package_version )
            else:
                raise ValueError( f"Package name didn't match the expected pattern : {package_fullname}" )

        print( "Loaded packages:", self.packages.keys() )
        print( "" )

        print( "Creating nodes" )

        # construct node graph data combining with already loaded package/asset data
        for node_elm in graph_json["nodeGraph"]["nodes"]:
            
            node_name = node_elm["name"]
            interface_fullname = node_elm["interface"]

            print( f"Processing {node_name}" )
            
            re_result = re.match( re_pattern_interface_fullname, interface_fullname )
            if re_result:
                account_id = re_result.group(1) # FIXME : check if this matches actual account id.
                package_name = re_result.group(2)
                interface_name = re_result.group(3)

                if account_id == "panorama":
                    if package_name=="abstract_rtsp_media_source":
                        pass
                    elif package_name=="hdmi_data_sink":
                        pass
                    else:
                        raise ValueError( f"Unsupported stock package name : {package_name}" )
                    
                else:
                    # FIXME : check if this matches actual account id.
                    pass
                    
                interface_elm = self.lookup_interface_from_package( package_name, interface_name )
                
                interface_category = interface_elm["category"]
                interface_asset_name = interface_elm["asset"]
                
                print( "package_name:", package_name )
                print( "interface_name:", interface_name )
                print( "interface_category:", interface_category )
                print( "interface_asset_name:", interface_asset_name )

                try:
                    asset_elm = self.lookup_asset_from_package( package_name, interface_asset_name )
                
                except KeyError as e:
                
                    if interface_category == "business_logic":
                        # In test-utility, we don't require asset for business logic. Use default information if missing.
                        asset_elm = {
                            "name": "code",
                            "implementations": [
                                {
                                    "type": "container",
                                    "assetUri": "",
                                    "descriptorUri": ""
                                }
                            ]
                        }
                    else:
                        raise

                asset_implementation_elm = asset_elm["implementations"][0] # FIXME : assuming "implementations" is always length=1
                asset_implementation_type = asset_implementation_elm["type"]
                
                if interface_category=="business_logic":

                    if asset_implementation_type == "container":
                        print( "Creating BusinessLogicContainerNode:", node_name )
                        node = BusinessLogicContainerNode( interface_elm, asset_elm )

                        if self.business_logic_node:
                            raise ValueError( "Multiple business logic nodes are not supported" )
                        self.business_logic_node = node

                        self.nodes[ node_name ] = node
                    else:
                        raise ValueError( f"Unsupported asset type '{asset_implementation_type}' for interface category '{interface_category}'" )
                    
                elif interface_category=="ml_model":

                    if asset_implementation_type == "model":
                        print( "Creating ModelNode:", node_name )
                        node = ModelNode( interface_elm, asset_elm )
                        self.nodes[ node_name ] = node
                    else:
                        raise ValueError( f"Unsupported asset type '{asset_implementation_type}' for interface category '{interface_category}'" )

                elif interface_category=="media_source":
                
                    print("asset_implementation_type:", asset_implementation_type)

                    if asset_implementation_type == "system":
                        asset_implementation_uri = asset_implementation_elm["assetUri"]
                        if asset_implementation_uri == "source/video/camera/rtsp/source_rtsp":
                            print( "Creating MediaSourceRtspCameraNode:", node_name )
                            node = MediaSourceRtspCameraNode( interface_elm, asset_elm )
                            self.nodes[ node_name ] = node
                        else:
                            raise ValueError( f"Unsupported asset uri '{asset_implementation_uri}' for asset implementation type '{asset_implementation_type}'" )
                    else:
                        raise ValueError( f"Unsupported asset type '{asset_implementation_type}' for interface category '{interface_category}'" )

                elif interface_category=="data_sink":
                    print( "Creating HdmiDataSinkNode:", node_name )
                    node = HdmiDataSinkNode( interface_elm, asset_elm )
                    self.nodes[ node_name ] = node
                else:
                    raise ValueError( f"Unknown interface category '{interface_category}'" )
            
            elif interface_fullname in ("boolean", "float32", "int32", "string"):

                print( "Creating ParameterNode:", node_name )
                node = ParameterNode( node_elm )
                self.nodes[ node_name ] = node
            
            else:
                raise ValueError( f"Interface name didn't match the expected pattern : {interface_fullname}" )

        print( "Created nodes:", self.nodes.keys() )
        print( "" )

        print( "Connecting edges" )

        # connect nodes using interfaces and edges
        for edge_elm in graph_json["nodeGraph"]["edges"]:
            print( "Resolving edge:", edge_elm )
            
            edge_producer = edge_elm["producer"]
            edge_consumer = edge_elm["consumer"]
            
            re_result = re.match( re_pattern_edge_fullname, edge_producer )
            if re_result:
                edge_producer_node_name = re_result.group(1)
                edge_producer_output_name = re_result.group(2)
            else:
                re_result = re.match( re_pattern_edge_parameter_node_name, edge_producer )
                if re_result:
                    edge_producer_node_name = re_result.group(1)
                    edge_producer_output_name = None
                else:
                    raise ValueError( f"Edge name didn't match the expected pattern : {edge_producer}" )

            re_result = re.match( re_pattern_edge_fullname, edge_consumer )
            if re_result:
                edge_consumer_node_name = re_result.group(1)
                edge_consumer_input_name = re_result.group(2)
            else:
                raise ValueError( f"Edge name didn't match the expected pattern : {edge_consumer}" )

            producer_node = self.nodes[edge_producer_node_name]
            consumer_node = self.nodes[edge_consumer_node_name]
            
            if isinstance( consumer_node, BusinessLogicContainerNode ):
                consumer_node.connect_producer( edge_consumer_input_name, producer_node, edge_producer_output_name )
            elif isinstance( producer_node, BusinessLogicContainerNode ):
                producer_node.connect_consumer( edge_producer_output_name, consumer_node, edge_consumer_input_name )
        
        print( "Inputs/Outputs of business logic container:" )
        print( "Inputs:", self.business_logic_node.inputs )
        print( "Outputs:", self.business_logic_node.outputs )
        

    def load_package_from_json( self, account_id, package_name, package_version ):

        package_dir = os.path.join( self.app_dir_top, "packages", f"{account_id}-{package_name}-{package_version}" )
        package_filepath = os.path.join( package_dir, "package.json" )

        print( "Loading package:", package_filepath )

        package = JsonPackage( package_filepath )
        #package.dump()
        
        # name fied in package.json is optional. check if it is same as graph.json if exists.
        if "name" in package.d["nodePackage"]:
            package_name_in_package = package.d["nodePackage"]["name"]
            if package_name_in_package != package_name:
                raise ValueError( f"Package name doesn't match : {package_name} != {package_name_in_package}" )

        # version fied in package.json is optional. check if it is same as graph.json if exists.
        if "version" in package.d["nodePackage"]:
            package_version_in_package = package.d["nodePackage"]["version"]
            if package_version_in_package != package_version:
                raise ValueError( f"Package version doesn't match : {package_version} != {package_version_in_package}" )

        self.packages[package_name] = package

    def lookup_interface_from_package( self, package_name, interface_name ):
        for interface_elm in self.packages[package_name].d["nodePackage"]["interfaces"]:
            if interface_elm["name"] == interface_name:
                return interface_elm
        raise KeyError( f"Interface '{interface_name}' not found in package '{package_name}'" )

    def lookup_asset_from_package( self, package_name, asset_name ):
        for asset_elm in self.packages[package_name].d["nodePackage"]["assets"]:
            if asset_elm["name"] == asset_name:
                return asset_elm
        raise KeyError( f"Asset '{asset_name}' not found in package '{package_name}'" )

    

# ---

# for testing
if 0:
    graph = Graph()

    graph.load(
        app_dir_top = "./pose_estimation_app",
        app_name = "pose_estimation_app",
    )

