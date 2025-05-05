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
        self.author_organization = ''
        self.organization = ''
        self.version = ''
        self.person_given_name = ''
        self.person_family_name = ''
        self.site_latitude = ''
        self.site_longitude = ''
        self.site_elevation = ''

        # Create a new IFC file and add header data
        self.ifc_file = ifcopenshell.file()
        self.ifc_file.header.file_description.description = ('ViewDefinition [DesignTransferView_V1.0]',)
        self.ifc_file.header.file_name.name = self.output_file
        self.ifc_file.header.file_name.time_stamp = ifcopenshell.util.date.datetime2ifc(datetime.datetime.now(), "IfcDateTime")
        self.ifc_file.header.file_name.author = (self.author_name,)
        self.ifc_file.header.file_name.organization = (self.author_organization,)
        self.ifc_file.header.file_name.preprocessor_version = 'IfcOpenShell {0}'.format(ifcopenshell.version)
        self.ifc_file.header.file_name.originating_system = 'CTU in Prague - Cloud2BIM - 1.1'
        self.ifc_file.header.file_name.authorization = 'None'

    @staticmethod
    def generate_guid():
        return ifcopenshell.guid.new()

    def create_local_placement(self, coordinates, axis=None, ref_direction=None, relative_to=None):
        """
        Creates an IfcLocalPlacement using a generic IfcAxis2Placement3D.
        :param coordinates: Tuple (x, y, z) for the location.
        :param axis: Optional IfcDirection entity for the axis.
        :param ref_direction: Optional IfcDirection entity for the reference direction.
        :param relative_to: Optional IfcLocalPlacement to relate to.
        """
        axis_placement = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=coordinates),
            Axis=axis,
            RefDirection=ref_direction
        )
        if relative_to:
            return self.ifc_file.create_entity("IfcLocalPlacement", RelativePlacement=axis_placement, PlacementRelTo=relative_to)
        else:
            return self.ifc_file.create_entity("IfcLocalPlacement", RelativePlacement=axis_placement)

    def create_extruded_solid(self, swept_area, position, extrusion_direction, depth):
        """
        Creates an IfcExtrudedAreaSolid entity.
        :param swept_area: The area profile to be extruded.
        :param position: The placement for the extrusion.
        :param extrusion_direction: The extrusion direction (IfcDirection).
        :param depth: The extrusion depth.
        """
        return self.ifc_file.create_entity(
            "IfcExtrudedAreaSolid",
            SweptArea=swept_area,
            Position=position,
            ExtrudedDirection=extrusion_direction,
            Depth=depth
        )

    def create_shape_representation(self, context, rep_id, rep_type, items):
        """
        Wraps geometry items into an IfcShapeRepresentation.
        :param context: The geometric representation context.
        :param rep_id: Representation Identifier (e.g., "Body", "Axis").
        :param rep_type: Representation type (e.g., "SweptSolid").
        :param items: List of geometry items.
        """
        return self.ifc_file.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=context,
            RepresentationIdentifier=rep_id,
            RepresentationType=rep_type,
            Items=items
        )

    def create_rel_defines_by_type(self, related_object, relating_type, description=None, name=None):
        """
        Creates an IfcRelDefinesByType entity
        :param related_object: The IFC object instance (e.g., wall, column, window).
        :param relating_type: The IFC type entity associated with the object (e.g., WallType, ColumnType).
        :param description: A textual description of the relationship.
        :param name: A name for the relationship.
        """
        return self.ifc_file.create_entity(
            "IfcRelDefinesByType",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=name,
            Description=description,
            RelatedObjects=[related_object],
            RelatingType=relating_type
        )

    def define_author_information(self, author_name, author_organization):
        self.author_name = author_name
        self.author_organization = author_organization
        self.ifc_file.header.file_name.author = (self.author_name,)
        self.ifc_file.header.file_name.organization = (self.author_organization,)

    def create_unit_assignment(self):
        length_unit = self.ifc_file.create_entity("IfcSIUnit", UnitType="LENGTHUNIT", Name="METRE")
        area_unit = self.ifc_file.create_entity("IfcSIUnit", UnitType="AREAUNIT", Name="SQUARE_METRE")
        volume_unit = self.ifc_file.create_entity("IfcSIUnit", UnitType="VOLUMEUNIT", Name="CUBIC_METRE")
        plane_angle_unit = self.ifc_file.create_entity("IfcSIUnit", UnitType="PLANEANGLEUNIT", Name="RADIAN")
        solid_angle_unit = self.ifc_file.create_entity("IfcSIUnit", UnitType="SOLIDANGLEUNIT", Name="STERADIAN")
        unit_assignment = self.ifc_file.create_entity(
            "IfcUnitAssignment",
            Units=[length_unit, area_unit, volume_unit, plane_angle_unit, solid_angle_unit]
        )
        return unit_assignment

    def create_material_with_color(self, name, rgb_values, transparency=0):
        """
        Create a complete material with color in one function call

        Args:
            name: Name of the material
            rgb_values: Tuple/list of (red, green, blue) values (0-1)
            transparency: Transparency value (0-1)

        Returns:
            tuple: (material, material_definition_representation)
        """
        # Create color and style objects
        color_rgb = self.ifc_file.create_entity(
            "IfcColourRgb",
            Name=None,
            Red=rgb_values[0],
            Green=rgb_values[1],
            Blue=rgb_values[2]
        )

        # Create surface style rendering
        surface_style_rendering = self.ifc_file.create_entity(
            "IfcSurfaceStyleRendering",
            SurfaceColour=color_rgb,
            Transparency=transparency,
            DiffuseColour=self.ifc_file.createIfcNormalisedRatioMeasure(0.4),
            TransmissionColour=None,
            DiffuseTransmissionColour=None,
            ReflectionColour=None,
            SpecularColour=None,
            SpecularHighlight=None,
            ReflectanceMethod="NOTDEFINED"
        )

        # Create surface style
        surface_style = self.ifc_file.create_entity(
            "IfcSurfaceStyle",
            Name=name,
            Side="BOTH",
            Styles=[surface_style_rendering]
        )

        # Create representation style assignment
        representation_style_assignment = self.ifc_file.create_entity(
            "IfcPresentationStyleAssignment",
            Styles=[surface_style]
        )

        # Create styled item
        styled_item = self.ifc_file.create_entity(
            "IfcStyledItem",
            Item=None,
            Styles=[representation_style_assignment],
            Name=None
        )

        # Create styled representation
        styled_representation = self.ifc_file.create_entity(
            "IfcStyledRepresentation",
            ContextOfItems=self.geom_rep_sub_context,
            RepresentationIdentifier="Style",
            RepresentationType="Material",
            Items=[styled_item]
        )

        # Create material
        material = self.ifc_file.create_entity(
            "IfcMaterial",
            Name=name,
            Description=None,
            Category=None
        )

        # Create material definition representation
        material_def_rep = self.ifc_file.create_entity(
            "IfcMaterialDefinitionRepresentation",
            Name='Representation',
            Description='Material Definition Representation',
            Representations=[styled_representation],
            RepresentedMaterial=material
        )

        return material, material_def_rep

    def assign_material(self, product, material):
        associated_material = self.ifc_file.create_entity(
            "IfcRelAssociatesMaterial",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
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

        unit_assignment = self.create_unit_assignment()

        axis_placement = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
            Axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            RefDirection=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )

        self.context = self.ifc_file.create_entity(
            "IfcGeometricRepresentationContext",
            ContextIdentifier="Body",
            ContextType="Model",
            CoordinateSpaceDimension=3,
            Precision=0.0001,
            WorldCoordinateSystem=axis_placement

        )

        self.geom_rep_sub_context = self.ifc_file.create_entity(
            "IfcGeometricRepresentationSubContext",
            ParentContext=self.context,
            ContextIdentifier="Body",
            ContextType="Model",
            TargetScale=None,
            TargetView="MODEL_VIEW",
            UserDefinedTargetView=None
        )

        self.geom_rep_sub_context_walls = self.ifc_file.create_entity(
            "IfcGeometricRepresentationSubContext",
            ParentContext=self.context,
            ContextIdentifier='Axis',
            ContextType="Model",
            TargetScale=None,
            TargetView="MODEL_VIEW",
            UserDefinedTargetView=None
        )

        self.project = self.ifc_file.create_entity(
            "IfcProject",
            GlobalId=self.generate_guid(),
            Name=self.project_name,
            LongName=self.long_project_name,
            ObjectType=self.object_type,
            Description=self.project_description,
            Phase=self.construction_phase,
            RepresentationContexts=[self.context],
            UnitsInContext=unit_assignment,
        )

        self.organization_entity = self.ifc_file.create_entity("IfcOrganization",
                                                               Name=self.organization
                                                               )

        self.application = self.ifc_file.create_entity("IfcApplication",
                                                       ApplicationDeveloper=self.organization_entity,
                                                       Version=self.version,
                                                       ApplicationFullName=self.project_name,
                                                       ApplicationIdentifier="Cloud2BIM"
                                                       )

        self.person_entity = self.ifc_file.create_entity("IfcPerson",
                                                         FamilyName=self.person_family_name,
                                                         GivenName=self.person_given_name
                                                         )

        self.person_and_organization_entity = self.ifc_file.create_entity("IfcPersonAndOrganization",
                                                                          ThePerson=self.person_entity,
                                                                          TheOrganization=self.ifc_file.by_type("IfcOrganization")[0]
                                                                          )

        self.owner_history = self.ifc_file.create_entity("IfcOwnerHistory",
                                                         OwningUser=self.person_and_organization_entity,
                                                         OwningApplication=self.application,
                                                         ChangeAction="NOTDEFINED",
                                                         CreationDate=ifcopenshell.util.date.datetime2ifc(datetime.datetime.now(), "IfcTimeStamp"),
                                                         )

        self.site = self.ifc_file.create_entity("IfcSite",
                                                GlobalId=self.generate_guid(),
                                                OwnerHistory=self.owner_history,
                                                Name="Site",
                                                CompositionType="ELEMENT",
                                                RefLatitude=self.site_latitude,
                                                RefLongitude=self.site_longitude,
                                                RefElevation=self.site_elevation
                                                )

        self.rel_aggregates_project = self.ifc_file.createIfcRelAggregates(
            ifcopenshell.guid.compress(uuid.uuid1().hex),
            self.owner_history,
            "$",
            "$",
            self.project,
            (self.site,)
        )

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

        self.building = self.ifc_file.create_entity("IfcBuilding",
                                                    GlobalId=self.generate_guid(),
                                                    OwnerHistory=self.owner_history,
                                                    Name="Building",
                                                    ObjectType="IfcBuilding",
                                                    ObjectPlacement=building_placement,
                                                    CompositionType="ELEMENT",
                                                    ElevationOfRefHeight=self.site_elevation
                                                    )

        self.ifc_file.create_entity("IfcRelAggregates",
                                    GlobalId=self.generate_guid(),
                                    OwnerHistory=self.owner_history,
                                    RelatingObject=self.site,
                                    RelatedObjects=[self.building]
                                    )

    def create_building_storey(self, storey_name, storey_elevation):
        axis_placement_storey = self.create_local_placement(
            (0.0, 0.0, float(storey_elevation)),
            axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            ref_direction=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )
        storey_placement = axis_placement_storey
        building_storey = self.ifc_file.create_entity("IfcBuildingStorey",
                                                      GlobalId=self.generate_guid(),
                                                      OwnerHistory=self.owner_history,
                                                      Name=storey_name,
                                                      Elevation=storey_elevation,
                                                      CompositionType="ELEMENT",
                                                      ObjectPlacement=storey_placement
                                                      )
        self.ifc_file.create_entity("IfcRelAggregates",
                                    GlobalId=self.generate_guid(),
                                    OwnerHistory=self.owner_history,
                                    RelatingObject=self.building,
                                    RelatedObjects=[building_storey]
                                    )
        return building_storey

    def create_slab(self, slab_name, points, slab_z_position, slab_height, material_name):
        # Process points (removing duplicates and ensuring float conversion) remains as before.
        polygon_points = [
            self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=point)
            for point in points
        ]
        # Close the polygon if needed
        if polygon_points[0].Coordinates != polygon_points[-1].Coordinates:
            polygon_points.append(polygon_points[0])
        polyline_profile = self.ifc_file.create_entity(
            "IfcArbitraryClosedProfileDef",
            ProfileType="AREA",
            ProfileName="Slab perimeter",
            OuterCurve=self.ifc_file.create_entity("IfcPolyline", Points=polygon_points)
        )

        slab_placement = self.create_local_placement(
            (0.0, 0.0, float(slab_z_position)),
            axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            ref_direction=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )
        # create an IfcAxis2Placement3D directly for the extrusion.
        axis_placement_extrusion = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
            Axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            RefDirection=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )
        slab_extrusion_direction = self.ifc_file.create_entity(
            "IfcDirection",
            DirectionRatios=(0.0, 0.0, 1.0)
        )

        slab_extrusion = self.create_extruded_solid(polyline_profile, axis_placement_extrusion, slab_extrusion_direction,
                                                    slab_height)
        shape_rep = self.create_shape_representation(self.geom_rep_sub_context, "Body", "SweptSolid", [slab_extrusion])
        slab = self.ifc_file.create_entity(
            "IfcSlab",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=slab_name,
            ObjectType="base",
            ObjectPlacement=slab_placement,
            Representation=self.ifc_file.create_entity(
                "IfcProductDefinitionShape",
                Representations=[shape_rep]
            ),
        )
        material = self.ifc_file.create_entity(
            "IfcMaterial",
            Name=material_name
        )
        self.ifc_file.create_entity(
            "IfcRelAssociatesMaterial",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            RelatedObjects=[slab],
            RelatingMaterial=material
        )
        return slab

    def assign_product_to_storey(self, product, storey):
        product_name = product.Name
        self.ifc_file.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=product_name,
            Description="Storey container for elements",
            RelatedElements=[product],
            RelatingStructure=storey
        )

    def create_material_layer(self, wall_thickness=0.3, material_name="Concrete"):
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
        wall_thickness_mm = wall_thickness * 1000
        material_layer_set = self.ifc_file.create_entity(
            "IfcMaterialLayerSet",
            MaterialLayers=material_layers,
            LayerSetName='Concrete load-bearing wall - %d mm' % wall_thickness_mm
        )
        return material_layer_set

    def create_material_layer_set_usage(self, material_layer_set, wall_thickness):
        material_layer_set_usage = self.ifc_file.create_entity(
            "IfcMaterialLayerSetUsage",
            ForLayerSet=material_layer_set,
            LayerSetDirection='AXIS2',
            DirectionSense='POSITIVE',
            OffsetFromReferenceLine=-(wall_thickness / 2)
        )
        return material_layer_set_usage

    def wall_placement(self, z_placement):
        return self.create_local_placement((0.0, 0.0, float(z_placement)))

    def wall_axis_placement(self, start_point=(0.0, 0.0), end_point=(5.0, 0.0)):
        start_cartesian_point = self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=start_point)
        end_cartesian_point = self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=end_point)
        wall_axis_polyline = self.ifc_file.create_entity(
            "IfcPolyline",
            Points=[start_cartesian_point, end_cartesian_point]
        )
        return wall_axis_polyline

    def wall_axis_representation(self, wall_axis_polyline):
        wall_axis_representation = self.ifc_file.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=self.geom_rep_sub_context_walls,
            RepresentationIdentifier="Axis",
            RepresentationType="Curve2D",
            Items=[wall_axis_polyline]
        )
        return wall_axis_representation

    def wall_swept_solid_representation(self, start_point, end_point, wall_height, wall_thickness):
        rectangle_reference_point = self.ifc_file.create_entity("IfcCartesianPoint",
                                                                Coordinates=((start_point[0] + end_point[0]) / 2, (start_point[1] + end_point[1]) / 2))
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
        rectangle_profile = self.ifc_file.create_entity(
            "IfcRectangleProfileDef",
            ProfileType='AREA',
            ProfileName='Wall Perimeter',
            Position=axis_placement_2d,
            XDim=math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2),
            YDim=wall_thickness,
        )

        wall_extruded_area = self.create_extruded_solid(
            rectangle_profile,
            position=None,
            extrusion_direction=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            depth=wall_height
        )
        wall_area_representation = self.create_shape_representation(
            self.geom_rep_sub_context_walls,
            'Body',
            'SweptSolid',
            [wall_extruded_area]
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
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name="Wall Name",
            Description="Wall Description",
            ObjectType="Wall",
            ObjectPlacement=wall_placement,
            Representation=product_definition_shape,
            Tag="Wall Tag",
            PredefinedType=None
        )
        return ifc_wall

    def create_wall_type(self, ifc_wall, wall_thickness=0.3):
        wall_thickness_mm = wall_thickness * 1000
        wall_type = self.ifc_file.create_entity(
            "IfcWallType",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name="Concrete 300",
            Description="Wall Load-bearing Concrete - thickness %d mm" % wall_thickness_mm,
            ApplicableOccurrence=None,
            HasPropertySets=None,
            RepresentationMaps=None,
            Tag="Wall Type Tag",
            ElementType="Wall Type",
            PredefinedType="SOLIDWALL"
        )
        rel_defines_by_type = self.create_rel_defines_by_type(ifc_wall, wall_type,
                                                              "Relation between Wall and WallType", None)
        rel_declares = self.ifc_file.create_entity(
            "IfcRelDeclares",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=None,
            Description=None,
            RelatingContext=self.project,
            RelatedDefinitions=[wall_type],
        )
        return wall_type, rel_defines_by_type, rel_declares

    def create_property_set(self, related_object, properties, name_local):
        property_set = self.ifc_file.create_entity(
            "IfcPropertySet",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=name_local,
            Description=None,
            HasProperties=[properties]
        )
        set_relation = self.ifc_file.create_entity(
            "IfcRelDefinesByProperties",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name="Cloud2BIM p_set",
            Description=None,
            RelatedObjects=[related_object],
            RelatingPropertyDefinition=property_set
        )
        return property_set, set_relation

    def create_property_single_value(self, property_name: str, boolean_value: bool):
        single_value = self.ifc_file.create_entity(
            "IfcPropertySingleValue",
            Name=property_name,
            Description=None,
            NominalValue=self.ifc_file.create_entity("IfcBoolean", boolean_value),
            Unit=None
        )

        return single_value

    def create_wall_opening(self, opening_placement, opening_representation):
        opening_standard_case = self.ifc_file.create_entity(
            "IfcOpeningElement",
            GlobalId=self.generate_guid(),
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

    def opening_placement(self, wall_start_point, wall_placement):
        axis = self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))  # Z-axis
        ref_direction = self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))  # X-axis

        axis_placement_window = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint",
                                                 Coordinates=(wall_start_point[0], wall_start_point[1], 0.0)),
            Axis = axis,
            RefDirection = ref_direction
        )
        window_placement = self.ifc_file.create_entity(
            "IfcLocalPlacement",
            PlacementRelTo=wall_placement,
            RelativePlacement=axis_placement_window
        )
        return axis_placement_window, window_placement

    def opening_representation(self, opening_extrusion_represent):
        opening_representation = self.ifc_file.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=self.context,
            RepresentationIdentifier='Body',
            RepresentationType='SweptSolid',
            Items=[opening_extrusion_represent],
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
        opening_extrusion = self.create_extruded_solid(
            polyline_profile_area,
            position=self.ifc_file.create_entity(
                "IfcAxis2Placement3D",
                Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(direction_x * offset_from_start,
                                                                                       direction_y * offset_from_start, opening_sill_height)),
                Axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
                RefDirection=self.ifc_file.createIfcDirection((direction_x, direction_y, 0.0))
            ),
            extrusion_direction=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            depth=opening_height
        )
        return opening_extrusion

    def create_rel_voids_element(self, relating_building_element, related_opening_element):
        rel_voids_element = self.ifc_file.create_entity(
            "IfcRelVoidsElement",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=None,
            Description=None,
            RelatingBuildingElement=relating_building_element,
            RelatedOpeningElement=related_opening_element
        )
        return rel_voids_element

    def create_rel_fills_element(self, opening_element, filling_element):
        rel_fills_element = self.ifc_file.create_entity(
            "IfcRelFillsElement",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=None,
            Description=None,
            RelatingOpeningElement=opening_element,
            RelatedBuildingElement=filling_element
        )
        return rel_fills_element

    def create_window(self, window_placement, product_definition_shape, window_id):
        window = self.ifc_file.create_entity(
            "IfcWindow",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name="Window Name",
            Description=window_id,
            ObjectType="Window",
            ObjectPlacement=window_placement,
            Representation=product_definition_shape,
            Tag="Window Tag",
            PredefinedType="NOTDEFINED"
        )
        return window

    def create_window_type(self):
        window_type_local = self.ifc_file.create_entity(
            "IfcWindowType",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name="Window (simple)",
            Description=None,
            ApplicableOccurrence=None,
            HasPropertySets=None,
            RepresentationMaps=None,
            Tag="Window Type Tag",
            ElementType=None,
            PredefinedType="NOTDEFINED"
        )
        return window_type_local

    def create_door(self, door_placement, product_definition_shape, door_id):
        door = self.ifc_file.create_entity(
            "IfcDoor",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name="Door Name",
            Description=door_id,
            ObjectType="Door",
            ObjectPlacement=door_placement,
            Representation=product_definition_shape,
            Tag="Door Tag",
            PredefinedType="NOTDEFINED"
        )
        return door

    def rel_contained_in_spatial_structure(self, ifc_element, ifc_building_storey):
        self.ifc_file.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=None,
            Description=None,
            RelatedElements=[ifc_element],
            RelatingStructure=ifc_building_storey
        )

    def space_placement(self, slab_z_position):
        return self.create_local_placement(
            (0.0, 0.0, float(slab_z_position)),
            axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            ref_direction=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )

    def create_space(self, dimensions, ifc_space_placement, floor_number, i, building_storey, extrusion_depth):
        context = self.geom_rep_sub_context
        points_polyline = []
        space_vertices = list(dimensions["vertices"])
        for vertex in space_vertices:
            point = self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(float(vertex[0]), float(vertex[1])))
            points_polyline.append(point)
        if points_polyline[0].Coordinates != points_polyline[-1].Coordinates:
            points_polyline.append(points_polyline[0])
        polyline = self.ifc_file.create_entity("IfcPolyline", Points=points_polyline)
        profile = self.ifc_file.create_entity(
            "IfcArbitraryClosedProfileDef",
            ProfileType="AREA",
            OuterCurve=polyline
        )
        extrusion_direction = self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        position = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
            Axis=None,
            RefDirection=None
        )
        solid = self.create_extruded_solid(profile, position, extrusion_direction, extrusion_depth)
        body_representation = self.create_shape_representation(context, "Body", "SweptSolid", [solid])
        product_definition_shape = self.ifc_file.create_entity(
            "IfcProductDefinitionShape",
            Representations=[body_representation]
        )
        ifc_space = self.ifc_file.create_entity(
            "IfcSpace",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=f"{str(floor_number) + '.' + str(i)}",
            Description=None,
            ObjectType=None,
            ObjectPlacement=ifc_space_placement,
            Representation=product_definition_shape,
            LongName=f"Room No. {str(floor_number) + '.' + str(i)} name",
            CompositionType="ELEMENT",
            PredefinedType="INTERNAL"
        )
        self.ifc_file.create_entity(
            "IfcRelContainedInSpatialStructure",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=None,
            Description=None,
            RelatedElements=[ifc_space],
            RelatingStructure=building_storey
        )
        return ifc_space

    def create_column_geometry_from_profile(self, type_name, point_coord_list, height):
        axis_placement = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity(
                "IfcCartesianPoint",
                Coordinates=(0.0, 0.0, 0.0)
            )
        )

        if type_name == 'round':
            radius = point_coord_list[0]  # Expecting: point_coord_list = [radius]
            profile = self.ifc_file.create_entity(
                "IfcCircleProfileDef",
                ProfileType="AREA",
                ProfileName="Circular",
                Position=self.ifc_file.create_entity(
                    "IfcAxis2Placement2D",
                    Location=self.ifc_file.create_entity(
                        "IfcCartesianPoint",
                        Coordinates=(0.0, 0.0)
                    )
                ),
                Radius=radius
            )
        else:
            curve = self.ifc_file.create_entity(
                "IfcIndexedPolyCurve",
                Points=self.ifc_file.create_entity(
                    "IfcCartesianPointList2d",
                    CoordList=point_coord_list
                ),
                Segments=[
                    self.ifc_file.create_entity(
                        "IfcLineIndex",
                        [*range(1, len(point_coord_list) + 1), 1]
                    )
                ],
                SelfIntersect=None
            )
            profile = self.ifc_file.create_entity(
                "IfcArbitraryClosedProfileDef",
                ProfileType="AREA",
                ProfileName=type_name,
                OuterCurve=curve
            )

        direction = self.ifc_file.create_entity(
            "IfcDirection",
            DirectionRatios=(0.0, 0.0, 1.0)
        )

        extruded_solid = self.create_extruded_solid(profile, axis_placement, direction, height)
        shape_rep = self.create_shape_representation(self.geom_rep_sub_context, "Body", "SweptSolid", [extruded_solid])
        product_definition_shape = self.ifc_file.create_entity(
            "IfcProductDefinitionShape",
            Name=None,
            Description=None,
            Representations=[shape_rep]
        )
        return product_definition_shape

    def create_column_type(self, type_name_local):
        column_type = self.ifc_file.create_entity(
            "IfcColumnType",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=type_name_local,
            Description=None,
            ApplicableOccurrence=None,
            HasPropertySets=None,
            RepresentationMaps=None,
            Tag=None,
            ElementType=None,
            PredefinedType="COLUMN"
        )
        return column_type

    def create_column_entity(self, column_id_local, placement, geometry):
        column = self.ifc_file.create_entity(
            "IfcColumn",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=column_id_local,
            Description=None,
            ObjectType=None,
            ObjectPlacement=placement,
            Representation=geometry,
            Tag=None,
            PredefinedType=None
        )
        return column

    def create_column(self, column_id, type_name, storey, placement_coords, vector_direction, points_2d, height):
        """
        Create IfcColumn
        :param column_id: Tag for identification (e.g., "C01", "R01", "ST01").
        :param type_name: Name of column type (e.g., "rect, round, steel").
        :param storey: Related storey ifcclass.
        :param placement_coords: coordinates for axis placement - centerpoint (x, y, z).
        :param vector_direction: rotation vector (0, 0, 0)
        :param points_2d: List of (x, y) tuples defining the closed polygon.
        :param height: Height of column.
        """
        # Placement
        ifc_axis = self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        rotation = self.ifc_file.create_entity("IfcDirection", DirectionRatios=vector_direction)
        column_placement = self.create_local_placement(placement_coords, axis= ifc_axis , ref_direction=rotation)

        # Geometry representation
        geometry = self.create_column_geometry_from_profile(type_name, points_2d, height)

        # Column entity
        column = self.create_column_entity(column_id, column_placement, geometry)

        # Column type
        column_type = self.create_column_type(type_name)

        # Relationships
        self.create_rel_defines_by_type(column, column_type)
        self.assign_product_to_storey(column, storey)

        return column

    def create_beam_type(self, type_name_local):
        """
        Creates an IfcBeamType entity with the specified name and default attributes.
        Parameters:
            type_name_local (str): The localized or descriptive name for the beam type.
        """
        beam_type = self.ifc_file.create_entity(
            "IfcBeamType",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=type_name_local,
            Description=None,
            ApplicableOccurrence=None,
            HasPropertySets=None,
            RepresentationMaps=None,
            Tag=None,
            ElementType=None,
            PredefinedType="BEAM"  # Valid enum in IfcBeamTypeEnum
        )
        return beam_type

    def create_beam_geometry(self, beam_type, length, point_list):
        """
        Create beam geometry of type 'rect' (rectangular) or 'steel' (I-shape), extruded along the X-axis.
        point_list: dimensions of the profile
        - for "rect": [width, height]
        - for "steel": list of 2D profile points [[x1,y1], [x2,y2], ...]
        """
        # Profile placement in 2D
        profile_position = self.ifc_file.create_entity(
            "IfcAxis2Placement2D",
            Location=self.ifc_file.create_entity(
                "IfcCartesianPoint",
                Coordinates=(0.0, 0.0)),
            RefDirection = self.ifc_file.create_entity(
                "IfcDirection",
                DirectionRatios=(1.0, 0.0)
            )
        )

        if beam_type == "rect":
            if point_list[0] is None or point_list[1] is None:
                raise ValueError("Rectangular beam requires 'width' and 'height'.")

            profile = self.ifc_file.create_entity(
                "IfcRectangleProfileDef",
                ProfileType="AREA",
                Position=profile_position,
                XDim=point_list[0],
                YDim=point_list[1]
            )

        elif beam_type == "steel":  # list of [x,y] coordinates
            curve = self.ifc_file.create_entity(
                "IfcIndexedPolyCurve",
                Points=self.ifc_file.create_entity(
                    "IfcCartesianPointList2D",
                    CoordList=point_list
                ),
                Segments=[
                    self.ifc_file.create_entity("IfcLineIndex", [*range(1, len(point_list) + 1), 1])
                ],
                SelfIntersect=None
            )
            profile = self.ifc_file.create_entity(
                "IfcArbitraryClosedProfileDef",
                ProfileType="AREA",
                ProfileName="SteelProfile",
                OuterCurve=curve
            )

        # Local 3D axis placement
        axis_placement = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint",Coordinates=(0.0, 0.0, 0.0)),
            Axis = self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 1.0, 0.0)),
            RefDirection = self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        )

        # Extrusion along X-axis
        direction = self.ifc_file.create_entity(
            "IfcDirection",
            DirectionRatios=(0.0, 0.0, 1.0)
        )

        # Create the solid and representation
        extruded_solid = self.create_extruded_solid(profile, axis_placement, direction, length)
        shape_rep = self.create_shape_representation(self.geom_rep_sub_context, "Body", "SweptSolid", [extruded_solid])

        product_definition_shape = self.ifc_file.create_entity(
            "IfcProductDefinitionShape",
            Name=None,
            Description=None,
            Representations=[shape_rep]
        )

        return product_definition_shape

    def create_beam_entity(self, beam_id_local, placement, geometry):
        """
        Creates a standard IfcBeam entity with minimal required attributes.
        :param beam_id_local: Name of the beam element (B01 , T001).
        :param placement: IfcLocalPlacement
        :param geometry: IfcProductDefinitionShape
        """
        beam = self.ifc_file.create_entity(
            "IfcBeam",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=beam_id_local,
            Description=None,
            ObjectType=None,
            ObjectPlacement=placement,
            Representation=geometry,
            Tag=None,
            PredefinedType="BEAM"
        )
        return beam

    def create_beam(self, beam_id, type_name, storey, placement_coords, vector_direction, points_2d, length, material):
        # Placement
        ifc_axis = self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0))
        rotation = self.ifc_file.create_entity("IfcDirection", DirectionRatios=vector_direction)
        beam_placement = self.create_local_placement(placement_coords, axis=ifc_axis, ref_direction=rotation)
        # Geometry representation
        beam_geometry = self.create_beam_geometry(type_name, length, points_2d)
        # beam entity
        beam = self.create_beam_entity(beam_id, beam_placement, beam_geometry)
        # beam type
        beam_type = self.create_beam_type(type_name)
        self.create_rel_defines_by_type(beam, beam_type)
        # Material
        self.assign_material(beam, material)
        # Relationships
        self.assign_product_to_storey(beam, storey)

    def create_stairs_entity(self, stairs_id_local):
        stairs = self.ifc_file.create_entity(
            "IfcStair",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=stairs_id_local,
            Description=None,
            ObjectType=None,
            ObjectPlacement=None,
            Representation=None,
            Tag=None,
            PredefinedType="NOTDEFINED"  # https://standards.buildingsmart.org/IFC/DEV/IFC4_2/FINAL/HTML/schema/ifcsharedbldgelements/lexical/ifcstairtypeenum.htm
        )
        return stairs

    def create_stair_flight(self, placement, geometry):
        stair_member = self.ifc_file.create_entity(
            "IfcStairFlight",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=None,
            ObjectPlacement=placement,
            Representation=geometry,
            PredefinedType="NOTDEFINED"
        )
        return stair_member

    def create_landing_slab(self, placement, geometry):
        landing = self.ifc_file.create_entity(
            "IfcSlab",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=None,
            ObjectPlacement=placement,
            Representation=geometry,
            PredefinedType="LANDING"
        )
        return landing

    def create_landing_slab_representation(self, points, thickness):
        polygon_points = [self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=point)
            for point in points]

        # Close the polygon if needed
        if polygon_points[0].Coordinates != polygon_points[-1].Coordinates:
            polygon_points.append(polygon_points[0])

        polyline_profile = self.ifc_file.create_entity(
            "IfcArbitraryClosedProfileDef",
            ProfileType="AREA",
            ProfileName="Slab perimeter",
            OuterCurve=self.ifc_file.create_entity("IfcPolyline", Points=polygon_points)
        )

        axis_placement_extrusion = self.ifc_file.create_entity(
            "IfcAxis2Placement3D",
            Location=self.ifc_file.create_entity("IfcCartesianPoint", Coordinates=(0.0, 0.0, 0.0)),
            Axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            RefDirection=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )

        slab_extrusion_direction = self.ifc_file.create_entity(
            "IfcDirection",
            DirectionRatios=(0.0, 0.0, 1.0)
        )

        slab_extrusion = self.create_extruded_solid(polyline_profile, axis_placement_extrusion,
                                                    slab_extrusion_direction, thickness)
        shape = self.create_shape_representation(self.geom_rep_sub_context, "Body", "SweptSolid", [slab_extrusion])
        shape_def = self.ifc_file.create_entity(
            "IfcProductDefinitionShape",
            Representations=[shape]
        )

        return shape_def

    def create_curved_stair_representation(self, number_of_raisers, raiser_height, angle_per_step_deg, inner_radius,
                                           step_width):
        vertices = []
        vertex_index_map = {}
        faces = []
        used_edges = set()

        def add_vertex(coord):
            if coord in vertex_index_map:
                return vertex_index_map[coord]
            index = len(vertices)
            vertices.append(coord)
            vertex_index_map[coord] = index
            return index

        def add_face(face_indices):
            nonlocal faces, used_edges
            reversed_needed = any(
                (face_indices[i], face_indices[(i + 1) % len(face_indices)]) in used_edges
                for i in range(len(face_indices))
            )
            if reversed_needed:
                face_indices = list(reversed(face_indices))
            for i in range(len(face_indices)):
                a = face_indices[i]
                b = face_indices[(i + 1) % len(face_indices)]
                used_edges.add((a, b))
            faces.append(face_indices)

        angle_per_step_rad = math.radians(angle_per_step_deg)
        outer_radius = inner_radius + step_width

        for i in range(number_of_raisers):
            angle_start = i * angle_per_step_rad
            angle_end = (i + 1) * angle_per_step_rad
            z = i * raiser_height
            z_top = (i + 1) * raiser_height

            # Define bottom and top points on inner and outer arcs
            A = (inner_radius * math.cos(angle_start), inner_radius * math.sin(angle_start), z)
            B = (inner_radius * math.cos(angle_start), inner_radius * math.sin(angle_start), z_top)
            C = (outer_radius * math.cos(angle_start), outer_radius * math.sin(angle_start), z)
            D = (outer_radius * math.cos(angle_start), outer_radius * math.sin(angle_start), z_top)

            E = (inner_radius * math.cos(angle_end), inner_radius * math.sin(angle_end), z_top)
            F = (outer_radius * math.cos(angle_end), outer_radius * math.sin(angle_end), z_top)

            idx_A = add_vertex(A)
            idx_B = add_vertex(B)
            idx_C = add_vertex(C)
            idx_D = add_vertex(D)
            idx_E = add_vertex(E)
            idx_F = add_vertex(F)

            # Riser (vertical front)
            add_face([idx_A, idx_B, idx_D, idx_C])

            # Tread surface (horizontal top)
            add_face([idx_B, idx_E, idx_F, idx_D])

            if i == 0:
                first_inner_idx = idx_A
                first_outer_idx = idx_C
            if i == number_of_raisers - 1:
                last_inner_idx = idx_E
                last_outer_idx = idx_F

        # Side faces
        # Inner side face
        side_inner_indices = [idx for idx, coord in sorted(
            [(idx, vtx) for idx, vtx in enumerate(vertices) if
             math.isclose(math.hypot(vtx[0], vtx[1]), inner_radius, abs_tol=1e-4)],
            key=lambda x: x[1][2]
        )]
        add_face(side_inner_indices)

        # Outer side face
        side_outer_indices = [idx for idx, coord in sorted(
            [(idx, vtx) for idx, vtx in enumerate(vertices) if
             math.isclose(math.hypot(vtx[0], vtx[1]), outer_radius, abs_tol=1e-4)],
            key=lambda x: x[1][2]
        )]
        add_face(side_outer_indices)

        # Bottom face
        add_face([first_inner_idx, first_outer_idx, last_outer_idx, last_inner_idx])

        # Back vertical face (last riser)
        add_face([last_inner_idx, last_outer_idx, idx_F, idx_E])

        # Bottom closure face
        add_face([first_inner_idx, first_outer_idx, last_outer_idx, last_inner_idx])

        # IFC entities
        point_list = self.ifc_file.create_entity("IfcCartesianPointList3D", CoordList=vertices)

        indexed_faces = []
        for face in faces:
            one_based_face = [index + 1 for index in face]
            face_entity = self.ifc_file.create_entity("IfcIndexedPolygonalFace", CoordIndex=one_based_face)
            indexed_faces.append(face_entity)

        polygon_face_set = self.ifc_file.create_entity(
            "IfcPolygonalFaceSet",
            Coordinates=point_list,
            Closed=True,
            Faces=indexed_faces
        )

        tessellation = self.ifc_file.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=self.geom_rep_sub_context,
            RepresentationIdentifier="Body",
            RepresentationType="Tessellation",
            Items=[polygon_face_set]
        )

        shape = self.ifc_file.create_entity(
            "IfcProductDefinitionShape",
            Representations=[tessellation]
        )

        return shape

    def create_stair_member_representation(self, number_of_raisers, raiser_height, tread_length, flight_width):
        vertices = []
        vertex_index_map = {}
        faces = []

        # Tracks all used oriented edges (a -> b)
        used_edges = set()

        def add_vertex(coord):
            """Add vertex only if it doesn't exist yet, return its index."""
            if coord in vertex_index_map:
                return vertex_index_map[coord]
            index = len(vertices)
            vertices.append(coord)
            vertex_index_map[coord] = index
            return index

        def add_face(face_indices):
            """Add a face with edge orientation checking to avoid GEM001 violations."""
            nonlocal faces, used_edges

            def edge(a, b):
                return (a, b)

            # Check all oriented edges of the face
            reversed_needed = False
            for i in range(len(face_indices)):
                a = face_indices[i]
                b = face_indices[(i + 1) % len(face_indices)]
                if (a, b) in used_edges:
                    reversed_needed = True
                    break

            # Reverse if needed to maintain consistent orientation
            if reversed_needed:
                face_indices = list(reversed(face_indices))

            # Register the edges
            for i in range(len(face_indices)):
                a = face_indices[i]
                b = face_indices[(i + 1) % len(face_indices)]
                used_edges.add((a, b))

            faces.append(face_indices)

        # Step 1: Front risers and treads
        for i in range(number_of_raisers):
            x = i * tread_length
            z = i * raiser_height
            z_top = (i + 1) * raiser_height

            A = (x, 0.0, z)
            B = (x, 0.0, z_top)
            C = (x, flight_width, z)
            D = (x, flight_width, z_top)

            idx_A = add_vertex(A)
            idx_B = add_vertex(B)
            idx_C = add_vertex(C)
            idx_D = add_vertex(D)

            add_face([idx_A, idx_B, idx_D, idx_C])  # Front riser

            x_next = (i + 1) * tread_length
            E = (x_next, 0.0, z_top)
            F = (x_next, flight_width, z_top)

            idx_E = add_vertex(E)
            idx_F = add_vertex(F)

            add_face([idx_B, idx_E, idx_F, idx_D])  # Tread surface

            if i == 0:
                first_A_idx = idx_A
                first_C_idx = idx_C

            if i == number_of_raisers - 1:
                last_E_idx = idx_E
                last_F_idx = idx_F

        # Step 2: Left side panel
        X = (number_of_raisers * tread_length, 0.0, (number_of_raisers - 1) * raiser_height)
        Y = (tread_length, 0.0, 0.0)
        idx_X = add_vertex(X)
        idx_Y = add_vertex(Y)

        left_side_points = [
            (idx, coord) for idx, coord in enumerate(vertices)
            if coord[1] == 0.0 and coord not in [X, Y]
        ]
        left_face_indices = [idx for idx, _ in sorted(left_side_points, key=lambda item: item[1][2])]
        left_face_indices.extend([idx_X, idx_Y])
        add_face(left_face_indices)

        # Step 3: Right side panel
        X_r = (number_of_raisers * tread_length, flight_width, (number_of_raisers - 1) * raiser_height)
        Y_r = (tread_length, flight_width, 0.0)
        idx_X_r = add_vertex(X_r)
        idx_Y_r = add_vertex(Y_r)

        right_side_points = [
            (idx, coord) for idx, coord in enumerate(vertices)
            if coord[1] == flight_width and coord not in [X_r, Y_r]
        ]
        right_face_indices = [idx for idx, _ in sorted(right_side_points, key=lambda item: item[1][2])]
        right_face_indices.extend([idx_X_r, idx_Y_r])
        add_face(right_face_indices)

        # Step 4: Bottom face
        add_face([idx_Y, idx_Y_r, first_C_idx, first_A_idx])

        # Step 5: Back vertical face
        add_face([idx_X, idx_X_r, last_F_idx, last_E_idx])

        # Step 6: Back underside face
        add_face([idx_Y, idx_Y_r, idx_X_r, idx_X])

        # IFC Geometry
        point_list = self.ifc_file.create_entity("IfcCartesianPointList3D", CoordList=vertices)

        indexed_faces = []
        for face in faces:
            one_based_face = [index + 1 for index in face]
            face_entity = self.ifc_file.create_entity("IfcIndexedPolygonalFace", CoordIndex=one_based_face)
            indexed_faces.append(face_entity)

        polygon_face_set = self.ifc_file.create_entity(
            "IfcPolygonalFaceSet",
            Coordinates=point_list,
            Closed=True,
            Faces=indexed_faces
        )

        tessellation = self.ifc_file.create_entity(
            "IfcShapeRepresentation",
            ContextOfItems=self.geom_rep_sub_context,
            RepresentationIdentifier="Body",
            RepresentationType="Tessellation",
            Items=[polygon_face_set]
        )

        shape = self.ifc_file.create_entity(
            "IfcProductDefinitionShape",
            Representations=[tessellation]
        )

        return shape

    def relate_stair_parts(self, stair, parts):
        rel = self.ifc_file.create_entity(
            "IfcRelAggregates",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            RelatingObject=stair,
            RelatedObjects=parts
        )
        return rel

    def create_stair(self, stair_id, storey, stair_parts, material):
        # Create a local placement for each part based on the provided origin
        for part in stair_parts:
            part["placement"] = self.create_local_placement(coordinates=part["origin"])

        # Generate geometry for each part if none is provided
        for part in stair_parts:
            if part["key"] == "flight_straight":
                part["geometry_representation"] = self.create_stair_member_representation(part["num_risers"],
                                                                                          part["raiser_height"],
                                                                                          part["tread_length"],
                                                                                          part["flight_width"])
            elif part["key"] == "landing":
                part["geometry_representation"] = self.create_landing_slab_representation(part["points"],
                                                                                          part["thickness"])
            elif part["key"] == "flight_curved":
                part["geometry_representation"] = self.create_curved_stair_representation(part["num_risers"],
                                                                                          part["raiser_height"],
                                                                                          part["angle_per_step_deg"],
                                                                                          part["inner_radius"],
                                                                                          part["flight_width"])

        # Create the logical stair container (IfcStair)
        stair = self.create_stairs_entity(stair_id)

        # Create and collect individual stair parts (flights and landings)
        parts = []
        for part in stair_parts:
            if part["key"] == "flight_straight":
                flight = self.create_stair_flight(part["placement"], part["geometry_representation"])
                parts.append(flight)
            elif part["key"] == "landing":
                landing = self.create_landing_slab(part["placement"], part["geometry_representation"])
                parts.append(landing)
            elif part["key"] == "flight_curved":
                landing = self.create_stair_flight(part["placement"], part["geometry_representation"])
                parts.append(landing)

        # Create an aggregation relationship between the stair container and its parts
        self.relate_stair_parts(stair, parts)
        # Assign material to the stair
        for part in parts:
            self.assign_material(part, material)
        # Relate the stair to the storey
        self.assign_product_to_storey(stair, storey)

        return stair

    def write(self):
        self.ifc_file.write(self.output_file)