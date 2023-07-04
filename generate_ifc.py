import ifcopenshell
import ifcopenshell.geom
import ifcopenshell.util.date
import ifcopenshell.util.unit
import ifcopenshell.util.element
import ifcopenshell.util.placement
import datetime
import random
import uuid


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
            Precision=1E-6,
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
            Description="Building container for elements",
            RelatedElements=[product],
            RelatingStructure=storey
        )

    def write(self):
        self.ifc_file.write(self.output_file)
