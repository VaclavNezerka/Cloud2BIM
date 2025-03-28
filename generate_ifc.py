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
        self.ifc_file.header.file_description.description = ('ViewDefinition [DesignTransferView_V1.0]',)
        self.ifc_file.header.file_name.name = self.output_file
        self.ifc_file.header.file_name.time_stamp = ifcopenshell.util.date.datetime2ifc(datetime.datetime.now(), "IfcDateTime")
        self.ifc_file.header.file_name.author = (self.author_name,)
        self.ifc_file.header.file_name.organization = (self.author_organization,)
        self.ifc_file.header.file_name.preprocessor_version = 'IfcOpenShell {0}'.format(ifcopenshell.version)
        self.ifc_file.header.file_name.originating_system = 'Cloud2BIM'
        self.ifc_file.header.file_name.authorization = 'None'

    def generate_guid(self):
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
        # [MODIFIED] Use the new helper to create a placement for the slab
        slab_placement = self.create_local_placement(
            (0.0, 0.0, float(slab_z_position)),
            axis=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(0.0, 0.0, 1.0)),
            ref_direction=self.ifc_file.create_entity("IfcDirection", DirectionRatios=(1.0, 0.0, 0.0))
        )
        axis_placement_slab_for_extrusion = self.create_local_placement((0.0, 0.0, 0.0))
        slab_extrusion_direction = self.ifc_file.create_entity(
            "IfcDirection",
            DirectionRatios=(0.0, 0.0, 1.0)
        )
        # [MODIFIED] Create extruded solid using the helper
        slab_extrusion = self.create_extruded_solid(polyline_profile, axis_placement_slab_for_extrusion, slab_extrusion_direction, slab_height)
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
        wall_thickness_mm = wall_thickness * 1000
        material_layer_set = self.ifc_file.create_entity(
            "IfcMaterialLayerSet",
            MaterialLayers=material_layers,
            LayerSetName='Concrete loadbearing wall - %d mm' % wall_thickness_mm
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
            ProfileName='Wall Perim',
            Position=axis_placement_2d,
            XDim=math.sqrt((end_point[0] - start_point[0]) ** 2 + (end_point[1] - start_point[1]) ** 2),
            YDim=wall_thickness,
        )
        # [MODIFIED] Use helper to create extruded solid and shape representation
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
            PredefinedType="STANDARD"
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
            PredefinedType="STANDARD"
        )
        rel_defines_by_type = self.ifc_file.create_entity(
            "IfcRelDefinesByType",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=None,
            Description="Relation between Wall and WallType",
            RelatedObjects=[ifc_wall],
            RelatingType=wall_type,
        )
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
                Axis=None,
                RefDirection=self.ifc_file.createIfcDirection((direction_x, direction_y))
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

    def create_rel_fills_element(self, relating_building_element, related_opening_element):
        rel_fills_element = self.ifc_file.create_entity(
            "IfcRelFillsElement",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=None,
            Description=None,
            RelatingOpeningElement=related_opening_element,
            RelatedBuildingElement=relating_building_element
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

    def rel_defined_by_type(self, window, window_type):
        rel_defined_by_type = self.ifc_file.create_entity(
            "IfcRelDefinesByType",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            Name=None,
            Description=None,
            RelatedObjects=[window],
            RelatingType=window_type
        )
        return rel_defined_by_type

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
        # [MODIFIED] Use helper for space placement with standard axis & ref_direction.
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

        curve = self.ifc_file.create_entity(
            "IfcIndexedPolyCurve",
            Points=self.ifc_file.create_entity(
                "IfcCartesianPointList2d",
                CoordList=point_coord_list
                ),
            Segments=[self.ifc_file.create_entity(
                "IfcLineIndex",
                [*range(1, len(point_coord_list) + 1), 1]
                )],
            SelfIntersect=None
            )


        column_area = self.ifc_file.create_entity(
            "IfcArbitraryClosedProfileDef",
            ProfileType="AREA",
            ProfileName=type_name,
            OuterCurve=curve
        )

        direction = self.ifc_file.create_entity(
            "IfcDirection",
            DirectionRatios=(0.0, 0.0, 1.0)
        )
        extruded_column_profile = self.create_extruded_solid(column_area, axis_placement, direction, height)
        shape_rep = self.create_shape_representation(self.geom_rep_sub_context, "Body", "SweptSolid", [extruded_column_profile])
        product_definition_shape = self.ifc_file.create_entity(
            "IfcProductDefinitionShape",
            Name=None,
            Description=None,
            Representations=[shape_rep]
        )
        return product_definition_shape

    def create_column_material_profile(self, material_name):
        material = self.ifc_file.create_entity("IfcMaterial", Name=material_name)
        profile = self.ifc_file.create_entity(
            "IfcMaterialProfile",
            Name="Column Profile",
            Material=material,
            Profile=None
        )
        profile_set = self.ifc_file.create_entity(
            "IfcMaterialProfileSet",
            Name="Column Profile Set",
            MaterialProfiles=[profile]
        )
        return profile_set

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

    def assign_material_to_column(self, column, material_profile_set):
        self.ifc_file.create_entity(
            "IfcRelAssociatesMaterial",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            RelatedObjects=[column],
            RelatingMaterial=material_profile_set
        )

    def define_column_type_relationship(self, column, column_type):
        self.ifc_file.create_entity(
            "IfcRelDefinesByType",
            GlobalId=self.generate_guid(),
            OwnerHistory=self.owner_history,
            RelatedObjects=[column],
            RelatingType=column_type
        )

    def create_column(self, column_id, type_name, storey, placement_coords, vector_direction, points_2d, height):
        """
        Create IfcColumn
        :param column_id: Tag for identification (e.g., "C01", "R01", "ST01").
        :param type_name: Name of column type (e.g., "Rectangular, Circular, Steel").
        :param storey: Related storey.
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
        self.define_column_type_relationship(column, column_type)
        self.assign_product_to_storey(column, storey)

        return column

    def write(self):
        self.ifc_file.write(self.output_file)