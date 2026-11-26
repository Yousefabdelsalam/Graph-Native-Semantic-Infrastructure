"""OWL generation, SHACL shapes and graph validation with Semantica.

Semantica:    ``OntologyEngine`` (``to_owl``, ``to_shacl``, ``validate_graph``,
              ``from_data``) and ``RDFExporter``.
Application:  declares the domain vocabulary and corrects one 0.7.0 limitation:
              for a property with several allowed ranges the generated shape
              keeps only the **first** one, so valid data is reported invalid.
              :func:`with_union_ranges` rewrites those constraints to ``sh:or``.
"""

from __future__ import annotations

from rdflib import BNode, Graph, Namespace, URIRef
from rdflib.collection import Collection
from rdflib.namespace import SH
from semantica.export import RDFExporter
from semantica.ontology import OntologyEngine

from backend.ontology.schema import (
    ALL_ENTITY_TYPES,
    ALL_RELATION_BY_NAME,
    RELATION_BY_NAME,
    llm_endpoints,
)

# RDFExporter writes types and predicates in this namespace; using it for the
# ontology makes the generated shapes target the exported data directly.
NAMESPACE = "https://semantica.dev/ns#"


def endpoints(relation: str) -> tuple[list[str], list[str]]:
    """Types allowed by either extraction vocabulary, rule types listed first."""
    spec = RELATION_BY_NAME.get(relation)
    domain, allowed = llm_endpoints(relation)
    return (
        list(dict.fromkeys([*(spec.domain if spec else ()), *domain])),
        list(dict.fromkeys([*(spec.range if spec else ()), *allowed])),
    )


def domain_ontology() -> dict:
    classes = [
        {"name": name, "uri": f"{NAMESPACE}{name}", "description": f"{name} in the Acme domain"}
        for name in ALL_ENTITY_TYPES
    ]
    properties = [
        {
            "name": spec.name,
            "uri": f"{NAMESPACE}{spec.name}",
            "type": "object",
            "domain": endpoints(spec.name)[0],
            "range": endpoints(spec.name)[1],
            "description": spec.description,
        }
        for spec in ALL_RELATION_BY_NAME.values()
    ]
    # RDFExporter emits entity names as semantica:text.
    properties.append(
        {
            "name": "text",
            "uri": f"{NAMESPACE}text",
            "type": "data",
            "domain": list(ALL_ENTITY_TYPES),
            "range": "string",
            "required": True,
        }
    )
    return {
        "uri": NAMESPACE.rstrip("#"),
        "name": "AcmeKnowledgeOntology",
        "version": "1.0",
        "classes": classes,
        "properties": properties,
    }


class OntologyService:
    def __init__(self) -> None:
        self.engine = OntologyEngine(base_uri=NAMESPACE)
        self.ontology = domain_ontology()

    def owl(self) -> str:
        """OWL in Turtle.

        Domains and ranges are passed as absolute IRIs: given plain class names,
        the 0.7.0 writer mints them in a different namespace from the declared
        class IRIs, leaving them pointing at undeclared classes.
        """
        absolute = {
            **self.ontology,
            "properties": [
                {
                    **prop,
                    "domain": [NAMESPACE + name for name in prop["domain"]],
                    "range": "xsd:string"
                    if prop["type"] == "data"
                    else [NAMESPACE + name for name in prop["range"]],
                }
                for prop in self.ontology["properties"]
            ],
        }
        return str(self.engine.to_owl(absolute, format="turtle"))

    def shacl(self) -> str:
        """Shapes exactly as Semantica generates them."""
        return self.engine.to_shacl(self.ontology)

    def corrected_shacl(self) -> str:
        return with_union_ranges(self.shacl())

    def graph_as_rdf(self, graph: dict[str, list[dict]]) -> str:
        return RDFExporter().export_to_rdf(graph, format="turtle")

    def validate(self, graph: dict[str, list[dict]], shapes: str) -> dict:
        report = self.engine.validate_graph(self.graph_as_rdf(graph), shacl=shapes)
        return {
            "conforms": report.conforms,
            "violations": [violation.to_dict() for violation in report.violations],
        }

    def infer_from_graph(self, graph: dict[str, list[dict]]) -> dict:
        """What Semantica would infer from the data, for comparison with the declared model."""
        return self.engine.from_data(graph)


def with_union_ranges(shapes_turtle: str) -> str:
    """Replace single ``sh:class`` constraints with ``sh:or`` over all declared ranges."""
    shapes = Graph().parse(data=shapes_turtle, format="turtle")
    namespace = Namespace(NAMESPACE)
    ranges = {
        namespace[name]: [namespace[r] for r in endpoints(name)[1]]
        for name in ALL_RELATION_BY_NAME
        if len(endpoints(name)[1]) > 1
    }
    for property_shape, path in list(shapes.subject_objects(SH.path)):
        if not isinstance(path, URIRef) or path not in ranges:
            continue
        if (property_shape, SH["class"], None) not in shapes:
            continue
        shapes.remove((property_shape, SH["class"], None))
        options = []
        for allowed in ranges[path]:
            option = BNode()
            shapes.add((option, SH["class"], allowed))
            options.append(option)
        head = BNode()
        Collection(shapes, head, options)
        shapes.add((property_shape, SH["or"], head))
    return shapes.serialize(format="turtle")
