import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.date
import ifcopenshell.util.unit
import ifcopenshell.util.element
import ifcopenshell.util.placement
import datetime
import uuid
import math


class IFCmodel:

    def __init__(self, project_name, output_file):
        self.project_name = project_name
        self.output_file = output_file
        self.project_description = ''
        self.object_type = ''
        self.long_project_name = ''
        self.construction_phase = ''
        self.author_name = ''
        self.author_organization = ''  # model author's organization
        self.organization = ''  # owner of the model or structure
        self.version = ''
        self.person_given_name = ''
        self.person_family_name = ''
        self.site_latitude = ''
        self.site_longitude = ''
        self.site_elevation = ''

        # Create a new IFC file and add header data
        self.ifc_file = ifcopenshell.file()
        self.ifc_file.header.file_description.description = ('ViewDefinition [DesignTransferView_V1.0]',)  # IFC schema subsets to describe data exchange for a specific use or workflow
        self.ifc_file.header.file_name.name = self.output_file
        self.ifc_file.header.file_name.time_stamp = ifcopenshell.util.date.datetime2ifc(datetime.datetime.now(), "IfcDateTime")
        self.ifc_file.header.file_name.author = (self.author_name,)
        self.ifc_file.header.file_name.organization = (self.author_organization,)
        self.ifc_file.header.file_name.preprocessor_version = 'IfcOpenShell {0}'.format(ifcopenshell.version)  # define the program used for IFC file creation
        self.ifc_file.header.file_name.originating_system = 'Cloud2BIM'
        self.ifc_file.header.file_name.authorization = 'None'

    def define_author_information(self, author_name, author_organization):
        self.author_name = author_name
        self.author_organization = author_organization
        self.ifc_file.header.file_name.author = (self.author_name,)
        self.ifc_file.header.file_name.organization = (self.author_organization,)

    def create_unit_assignment(self):
        """Create a unit assignment for the project."""
        # Define length, area, volume and angle units (SI units are used here)
        length_unit = self.ifc_file.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
        area_unit = self.ifc_file.create_entity("IfcSIUnit", UnitType="AREAUNIT", Name="SQUARE_METRE")
        volume_unit = self.ifc_file.create_entity("IfcSIUnit", UnitType="VOLUMEUNIT", Name="CUBIC_METRE")
        plane_angle_unit = self.ifc_file.create_entity("IfcSIUnit", UnitType="PLANEANGLEUNIT", Name="RADIAN")
        solid_angle_unit = self.ifc_file.create_entity("IfcSIUnit", UnitType="SOLIDANGLEUNIT", Name="STERADIAN")

        # Define a unit assignment with these units
        unit_assignment = self.ifc_file.create_entity(
            "IfcUnitAssignment",
            Units=[length_unit, area_unit, volume_unit, plane_angle_unit, solid_angle_unit]
        )

        return unit_assignment

    def assign_material(self, product, material):
        associated_material = self.ifc_file.create_entity(
            "IfcRelAssociatesMaterial",
            GlobalId=ifcopenshell.guid.new(),
            RelatingMaterial=material,
            RelatedObjects=[product]
        )
        return associated_material

    def define_project_data(self, project_description, object_type, long_project_name, construction_phase, version,
                            organization, person_given_name, person_family_name, latitude, longitude, elevation):
        self.project_description = project_description
        self.person_given_name = person_given_name
        self.person_family_name = person_family_name
        self.object_type = object_type
        self.long_project_name = long_project_name
        self.construction_phase = construction_phase
        self.version = version
        self.organization = organization
        self.site_latitude = latitude
        self.site_longitude = longitude
        self.site_elevation = elevation

        # Define a unit assignment
        unit_assignment = self.create_unit_assignment()

        # Inception of coordination system - related to World coordinate system
        axis_placement = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
            Axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            RefDirection=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )

        # Define geometric context (swept-solid objects)
        self.context = self.ifc_file.create_entity(
            "IfcGeometricRepresentationContext",
            ContextIdentifier="Body",
            ContextType="Model",
            CoordinateSpaceDimension=3,
            Precision=0.0001,
            WorldCoordinateSystem=axis_placement
        )

        # Create geometric representation sub-context (swept-solid objects)
        self.geom_rep_sub_context = self.ifc_file.create_entity(
            "IfcGeometricRepresentationSubContext",
            ParentContext=self.context,
            ContextIdentifier="Body",
            ContextType="Model",
            TargetScale=None,
            TargetView="MODEL_VIEW",
            UserDefinedTargetView=None
        )

        # Create geometric representation sub-context (swept-solid objects)
        self.geom_rep_sub_context_walls = self.ifc_file.create_entity(
            "IfcGeometricRepresentationSubContext",
            ParentContext=self.context,
            ContextIdentifier='Axis',
            ContextType="Model",
            TargetScale=None,
            TargetView="MODEL_VIEW",
            UserDefinedTargetView=None
        )

        # Set up the project structure
        self.project = self.ifc_file.create_entity(
            "IfcProject",
            GlobalId=ifcopenshell.guid.new(),
            Name=self.project_name,
            LongName=self.long_project_name,
            ObjectType=self.object_type,
            Description=self.project_description,
            Phase=self.construction_phase,
            UnitsInContext=unit_assignment,  # use the created unit assignment here
        )

        # Create the organization entity
        self.organization_entity = self.ifc_file.create_entity("IfcOrganization",
                                                               Name=self.organization
                                                               )

        # Set the application information
        self.application = self.ifc_file.create_entity("IfcApplication",
                                                       ApplicationDeveloper=self.organization_entity,
                                                       Version=self.version,
                                                       ApplicationFullName=self.project_name,
                                                       ApplicationIdentifier="MY_IFC_APP"
                                                       )

        # Create the person entity
        self.person_entity = self.ifc_file.create_entity("IfcPerson",
                                                         FamilyName=self.person_family_name,
                                                         GivenName=self.person_given_name
                                                         )

        # Create the person and organization entity
        self.person_and_organization_entity = self.ifc_file.create_entity("IfcPersonAndOrganization",
                                                                          ThePerson=self.person_entity,
                                                                          TheOrganization=self.ifc_file.by_type("IfcOrganization")[0]
                                                                          )

        # Create an owner history
        self.owner_history = self.ifc_file.create_entity("IfcOwnerHistory",
                                                         OwningUser=self.person_and_organization_entity,
                                                         OwningApplication=self.application,
                                                         ChangeAction="NOTDEFINED",
                                                         CreationDate=ifcopenshell.util.date.datetime2ifc(datetime.datetime.now(), "IfcTimeStamp"),
                                                         )

        # Create the site
        self.site = self.ifc_file.create_entity("IfcSite",
                                                GlobalId=ifcopenshell.guid.new(),
                                                OwnerHistory=self.owner_history,
                                                Name="Site",
                                                CompositionType="ELEMENT",
                                                RefLatitude=self.site_latitude,
                                                RefLongitude=self.site_longitude,
                                                RefElevation=self.site_elevation
                                                )

        # relationship between the IfcProject and IfcSite entities
        self.rel_aggregates_project = self.ifc_file.createIfcRelAggregates(
            ifcopenshell.guid.compress(uuid.uuid1().hex),
            self.owner_history,
            "$",
            "$",
            self.project,
            (self.site,)
        )

        # Inception of coordination system - related to building
        axis_placement_building = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
            Axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            RefDirection=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )

        building_placement = self.ifc_file.create_entity(
            "IfcLocalPlacement",
            RelativePlacement=axis_placement_building
        )

        # Create the building
        self.building = self.ifc_file.create_entity("IfcBuilding",
                                                    GlobalId=ifcopenshell.guid.new(),
                                                    OwnerHistory=self.owner_history,
                                                    Name="Building",
                                                    ObjectType="IfcBuilding",
                                                    ObjectPlacement=building_placement,
                                                    CompositionType="ELEMENT",
                                                    ElevationOfRefHeight=self.site_elevation
                                                    )

        # Create IfcRelAggregates entities to connect the site and the building
        self.ifc_file.create_entity("IfcRelAggregates",
                                    GlobalId=ifcopenshell.guid.new(),
                                    OwnerHistory=self.owner_history,
                                    RelatingObject=self.site,
                                    RelatedObjects=[self.building]
                                    )

    def create_building_storey(self, storey_name, storey_elevation):
        # Inception of coordination system - related to storey
        axis_placement_storey = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, float(storey_elevation))),
            Axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            RefDirection=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )

        storey_placement = self.ifc_file.create_entity(
            "IfcLocalPlacement",
            RelativePlacement=axis_placement_storey
        )

        # Create building storey
        building_storey = self.ifc_file.create_entity("IfcBuildingStorey",
                                                      GlobalId=ifcopenshell.guid.new(),
                                                      OwnerHistory=self.owner_history,
                                                      Name=storey_name,
                                                      Elevation=storey_elevation,
                                                      CompositionType="ELEMENT",
                                                      ObjectPlacement=storey_placement
                                                      )

        # relationship between IfcBuilding and IfcBuildingStorey
        self.ifc_file.create_entity("IfcRelAggregates",
                                    GlobalId=ifcopenshell.guid.new(),
                                    OwnerHistory=self.owner_history,
                                    RelatingObject=self.building,
                                    RelatedObjects=[building_storey]
                                    )
        return building_storey

    def create_slab(self, slab_name, points, slab_z_position, slab_height, material_name):
        # Convert points to IfcCartesianPoint instances
        polygon_points = [
            self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=point)
            for point in points
        ]

        # Inception of coordination system - related to slab
        axis_placement_slab = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, float(slab_z_position))),
            Axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            RefDirection=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )

        # Inception of coordination system - related to slab
        axis_placement_slab_for_extrusion = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
            Axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            RefDirection=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )

        slab_placement = self.ifc_file.create_entity(
            "IfcLocalPlacement",
            RelativePlacement=axis_placement_slab
        )

        slab_extrusion_direction = self.ifc_file.create_entity(
            "IfcDirection",
            DirectionRatios=(0.0, 0.0, 1.0)
        )

        polyline_profile = self.ifc_file.create_entity(
            "IfcArbitraryClosedProfileDef",
            ProfileType="AREA",
            ProfileName="Slab perimeter",
            OuterCurve=self.ifc_file.create_entity("IfcPolyline", Points=polygon_points + [polygon_points[0]])
        )

        slab_extrusion = self.ifc_file.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=polyline_profile,
            Position=axis_placement_slab_for_extrusion,
            ExtrudedDirection=slab_extrusion_direction,
            Depth=slab_height
        )

        # Create IfcSlab entity with slab_name
        slab = self.ifc_file.create_entity(
            "IfcSlab",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=self.owner_history,
            Name=slab_name,
            ObjectType="base",
            ObjectPlacement=slab_placement,
            Representation=self.ifc_file.create_entity(
                "IfcProductDefinitionShape",
                Representations=[
                    self.ifc_file.create_entity(
                        "IfcShapeRepresentation",
                        ContextOfItems=self.geom_rep_sub_context,
                        RepresentationIdentifier="Body",
                        RepresentationType="SweptSolid",
                        Items=[slab_extrusion],
                    )
                ],
            ),
        )

        # Create material
        material = self.ifc_file.create_entity(
            "IfcMaterial",
            Name=material_name
        )

        # Associate material to the slab
        self.ifc_file.create_entity(
            "IfcRelAssociatesMaterial",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=self.owner_history,
            RelatedObjects=[slab],
            RelatingMaterial=material
        )

        return slab

    def assign_product_to_storey(self, product, storey):
        product_name = product.Name
        self.ifc_file.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=self.owner_history,
            Name=product_name,
            Description="Storey container for elements",
            RelatedElements=[product],
            RelatingStructure=storey
        )

    # wall definition

    def create_material_layer(self, wall_thickness=0.3, material_name="Masonry - brick"):
        material_layer = self.ifc_file.create_entity(
            "IfcMaterialLayer",
            LayerThickness=wall_thickness,
            Name='Core',
            IsVentilated=".F.",
            Category='LoadBearing',
            Priority=99,
            Material=self.ifc_file.create_entity(
                "IfcMaterial",
                Name=material_name
            )
        )
        return material_layer

    def create_material_layer_set(self, material_layers=None, wall_thickness=0.3):
        wall_thickness = wall_thickness * 1000
        # Create an IfcMaterialLayerSet using the provided layers
        material_layer_set = self.ifc_file.create_entity(
            "IfcMaterialLayerSet",
            MaterialLayers=material_layers,
            LayerSetName='Concrete loadbearing wall - %d mm' % wall_thickness
        )

        return material_layer_set

    def create_material_layer_set_usage(self, material_layer_set, wall_thickness):
        # Create an IFCMaterialLayerSetUsage using the provided material layer set
        material_layer_set_usage = self.ifc_file.create_entity(
            "IfcMaterialLayerSetUsage",
            ForLayerSet=material_layer_set,
            LayerSetDirection='AXIS2',
            DirectionSense='POSITIVE',
            OffsetFromReferenceLine=-(wall_thickness / 2)  # Adjust the offset as needed
        )
        return material_layer_set_usage

    def wall_placement(self, z_placement):
        # Inception of coordination system - related to wall
        axis_placement_wall = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, z_placement)),
            Axis=None,
            RefDirection=None
        )

        wall_placement = self.ifc_file.create_entity(
            "IfcLocalPlacement",
            RelativePlacement=axis_placement_wall
        )
        return wall_placement

    def wall_axis_placement(self, start_point=(0.0, 0.0), end_point=(5.0, 0.0)):
        # Convert points to IfcCartesianPoint instances
        start_cartesian_point = self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=start_point)
        end_cartesian_point = self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=end_point)

        # Create an IfcPolyline with the points
        wall_axis_polyline = self.ifc_file.create_entity(
            "IfcPolyline",
            Points=[start_cartesian_point, end_cartesian_point]
        )
        return wall_axis_polyline

    def wall_axis_representation(self, wall_axis_polyline):
        # Create an IfcShapeRepresentation for the wall
        wall_axis_representation = self.ifc_file.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=self.geom_rep_sub_context_walls,  # Replace with the appropriate context
            RepresentationIdentifier="Axis",
            RepresentationType="Curve2D",  # Use "Curve" as per your desired output
            Items=[wall_axis_polyline],  # Replace with the appropriate geometry items for the wall = IfcPolyline
        )
        return wall_axis_representation

    def wall_swept_solid_representation(self, start_point, end_point, wall_height, wall_thickness):
        # Create an IfcCartesianPoint for the reference point of the rectangle (center or any other point)
        rectangle_reference_point = self.ifc_file.create_entity("IfcCartesianPoint",
                                                                Coordinates=((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2)
                                                                )

        # Create an IfcAxis2Placement2D using the center point
        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        magnitude = math.sqrt(dx ** 2 + dy ** 2)
        direction_x = float(dx / magnitude)
        direction_y = float(dy / magnitude)
        axis_placement_2d = self.ifc_file.create_entity(
            "IfcAxis2Placement2D",
            Location=rectangle_reference_point,
            RefDirection=self.ifc_file.createIfcDirection((direction_x, direction_y))
        )

        # Create an IfcRectangleProfileDef with the specified attributes
        rectangle_profile = self.ifc_file.create_entity(
            "IfcRectangleProfileDef",
            ProfileType='AREA',
            ProfileName='Wall Perim',
            Position=axis_placement_2d,
            XDim=math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2),
            YDim=wall_thickness,  # Replace with the actual Y dimension
        )

        # Create an IfcExtrudedAreaSolid
        wall_extruded_area = self.ifc_file.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=rectangle_profile,
            Position=None,
            ExtrudedDirection=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),  # direction of extrusion
            Depth=wall_height,  # Replace with the actual wall height
        )

        # Create an IfcShapeRepresentation for the wall
        wall_area_representation = self.ifc_file.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=self.geom_rep_sub_context_walls,  # Replace with the appropriate context
            RepresentationIdentifier='Body',
            RepresentationType='SweptSolid',
            Items=[wall_extruded_area],  # Replace with the appropriate geometry items for the wall
        )
        return rectangle_profile, wall_extruded_area, wall_area_representation

    def product_definition_shape(self, wall_axis_representation=None, wall_area_representation=None):
        product_definition_shape = self.ifc_file.create_entity(
            "IfcProductDefinitionShape",
            Representations=[wall_axis_representation, wall_area_representation[2]]
        )
        return product_definition_shape

    def create_wall(self, wall_placement, product_definition_shape):
        ifc_wall = self.ifc_file.create_entity(
            "IfcWall",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=self.owner_history,  # Replace with your IfcOwnerHistory entity or None
            Name="Wall Name",  # Replace with your wall's name or None
            Description="Wall Description",  # Replace with your wall's description or None
            ObjectType="Wall",
            ObjectPlacement=wall_placement,  # Replace with your IfcLocalPlacement or IfcGridPlacement entity or None
            Representation=product_definition_shape,  # Replace with your IfcProductDefinitionShape entity or None
            Tag="Wall Tag",  # Replace with your wall's tag or None
            PredefinedType="STANDARD"  # Replace with your wall's predefined type or None
        )
        return ifc_wall

    def create_wall_type(self, ifc_wall, wall_thickness=0.3):
        wall_thickness = wall_thickness * 1000
        wall_type = self.ifc_file.create_entity(
            "IfcWallType",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=self.owner_history,
            Name="Concrete 300",
            Description="Wall Load-bearing Concrete - thickness %d mm" % wall_thickness,
            ApplicableOccurrence=None,
            HasPropertySets=None,
            RepresentationMaps=None,  # Replace with your representation maps
            Tag="Wall Type Tag",  # Replace with your wall type's tag
            ElementType="Wall Type",  # A descriptive name for the element type
            PredefinedType="STANDARD"  # Replace with your wall type's predefined type
        )

        # Create the IfcRelDefinesByType relationship
        rel_defines_by_type = self.ifc_file.create_entity(
            "IfcRelDefinesByType",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=self.owner_history,
            Name=None,
            Description="Relation between Wall and WallType",
            RelatedObjects=[ifc_wall],
            RelatingType=wall_type,
        )

        rel_declares = self.ifc_file.create_entity(
            "IfcRelDeclares",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=self.owner_history,
            Name=None,
            Description=None,
            RelatingContext=self.project,
            RelatedDefinitions=[wall_type],
        )
        return wall_type, rel_defines_by_type, rel_declares

    # Wall opening definition

    def create_wall_opening(self, opening_placement, opening_representation):
        opening_standard_case = self.ifc_file.create_entity(
            "IfcOpeningElement",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=self.owner_history,
            Name="Opening ID",
            Description="Wall opening",
            ObjectType=None,
            ObjectPlacement=opening_placement,
            Representation=opening_representation,
            Tag=None,
            PredefinedType="OPENING",
        )
        return opening_standard_case

    # opening placement
    def opening_placement(self, wall_start_point, wall_placement):
        # Inception of coordination system - related to wall
        axis_placement_window = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(wall_start_point[0], wall_start_point[1], 0.0)),
            Axis=None,
            RefDirection=None
        )

        window_placement = self.ifc_file.create_entity(
            "IfcLocalPlacement",
            PlacementRelTo=wall_placement,
            RelativePlacement=axis_placement_window
        )
        return axis_placement_window, window_placement

    def opening_representation(self, opening_extrusion_represent):
        # Create an IfcShapeRepresentation for the opening
        opening_representation = self.ifc_file.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=self.context,  # Replace with the appropriate context
            RepresentationIdentifier='Body',
            RepresentationType='SweptSolid',
            Items=[opening_extrusion_represent],  # Replace with the appropriate geometry items for the opening
        )
        return opening_representation

    def product_definition_shape_opening(self, opening_representation):
        product_definition_shape = self.ifc_file.create_entity(
            "IfcProductDefinitionShape",
            Representations=[opening_representation]
        )
        return product_definition_shape

    def opening_closed_profile_def(self, opening_width, wall_thickness):

        points = [(0.0, - wall_thickness / 2), (0.0, wall_thickness/2), (opening_width, wall_thickness / 2), (opening_width, - wall_thickness/2)]
        points = [(float(x), float(y)) for x, y in points]

        # Convert points to IfcCartesianPoint instances
        extrusion_points = [
            self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=point)
            for point in points
        ]

        polyline_profile_area = self.ifc_file.create_entity(
            "IfcArbitraryClosedProfileDef",
            ProfileType="AREA",
            ProfileName="Opening perimeter",
            OuterCurve=self.ifc_file.create_entity("IfcPolyline", Points=extrusion_points + [extrusion_points[0]])
        )
        return polyline_profile_area

    def opening_extrusion(self, polyline_profile_area, opening_height, start_point, end_point, opening_sill_height, offset_from_start):

        dx = end_point[0] - start_point[0]
        dy = end_point[1] - start_point[1]
        magnitude = math.sqrt(dx ** 2 + dy ** 2)
        direction_x = float(dx / magnitude)
        direction_y = float(dy / magnitude)

        opening_extrusion = self.ifc_file.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=polyline_profile_area,
            Position=self.ifc_file.create_entity(
                "IfcAxis2Placement3D",
                Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(direction_x * offset_from_start,
                                                                                       direction_y * offset_from_start, opening_sill_height)),
                Axis=None,
                RefDirection=self.ifc_file.createIfcDirection((direction_x, direction_y))
            ),
            ExtrudedDirection=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            Depth=opening_height
        )
        return opening_extrusion

    def create_rel_voids_element(self, relating_building_element, related_opening_element):
        rel_voids_element = self.ifc_file.create_entity(
            "IfcRelVoidsElement",
            GlobalId=ifcopenshell.guid.new(),
            OwnerHistory=self.owner_history,
            Name=None,  # this corresponds to the "$" in your IFC code
            Description=None,  # this corresponds to the "$" in your IFC code
            RelatingBuildingElement=relating_building_element,
            RelatedOpeningElement=related_opening_element
        )
        return rel_voids_element

    def write(self):
        self.ifc_file.write(self.output_file)
