#!/usr/bin/env python3
import requests
import yaml
import base64
from lxml import etree
from dataclasses import dataclass, field
from pprint import pprint


@dataclass
class SearchRequest:
    query: str
    page: int
    sortby: str | None


@dataclass
class Passage:
    text: str


@dataclass
class Properties:
    lang: str | None

@dataclass
class Document:
    doc_id: str
    url: str
    domain: str
    title: str
    headline: str | None
    modtime: str | None
    size: int | None
    mime_type: str | None
    passages: list[Passage] = field(default_factory=list)
    properties: Properties | None = None

    
@dataclass
class Group:
    category: str
    doccount: int
    documents: list[Document]


@dataclass
class SearchResponse:
    reqid: str
    found: dict[str, int]
    found_human: str | None
    is_local: bool
    groups: list[Group]


@dataclass
class YandexSearchResult:
    request: SearchRequest
    response: SearchResponse



def parse_yandex_search(xml_bytes: bytes) -> YandexSearchResult:
    root = etree.fromstring(xml_bytes)

    # -------- request --------
    req_el = root.find("request")
    request = SearchRequest(
        query=req_el.findtext("query"),
        page=int(req_el.findtext("page")),
        sortby=req_el.findtext("sortby"),
    )

    # -------- response --------
    resp_el = root.find("response")

    found = {}
    for f in resp_el.findall("found"):
        found[f.attrib["priority"]] = int(f.text)

    groups: list[Group] = []

    for grouping in resp_el.findall(".//grouping"):
        for g in grouping.findall("group"):
            categ = g.find("categ").attrib["name"]
            doccount = int(g.findtext("doccount"))

            documents: list[Document] = []

            for d in g.findall("doc"):
                passages = [
                    Passage("".join(p.itertext()))
                    for p in d.findall(".//passage")
                ]

                props_el = d.find("properties")
                properties = None
                if props_el is not None:
                    properties = Properties(
                        lang=props_el.findtext("lang")
                    )

                documents.append(
                    Document(
                        doc_id=d.attrib["id"],
                        url=d.findtext("url"),
                        domain=d.findtext("domain"),
                        title="".join(d.find("title").itertext()),
                        headline=(
                            "".join(d.find("headline").itertext())
                            if d.find("headline") is not None
                            else None
                        ),
                        modtime=d.findtext("modtime"),
                        size=int(d.findtext("size")) if d.findtext("size") else None,
                        mime_type=d.findtext("mime-type"),
                        passages=passages,
                        properties=properties,
                    )
                )

            groups.append(
                Group(
                    category=categ,
                    doccount=doccount,
                    documents=documents,
                )
            )

    response = SearchResponse(
        reqid=resp_el.findtext("reqid"),
        found=found,
        found_human=resp_el.findtext("found-human"),
        is_local=resp_el.findtext("is-local") == "yes",
        groups=groups,
    )

    return YandexSearchResult(
        request=request,
        response=response,
    )




keys = yaml.load(open('key.yaml').read(), yaml.Loader)

res = requests.post("https://searchapi.api.cloud.yandex.net/v2/web/search", headers={
    "Authorization": f"Bearer {keys['secret']}"
}, json={
    "query": {
        "search_type": "SEARCH_TYPE_RU",
        "query_text": "Где купить шубу?"
    }
})

xml = base64.b64decode(res.json()['rawData'])
print(xml)

pprint(parse_yandex_search(xml))

