from __future__ import annotations

import json
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import streamlit as st
from lxml import etree
from pyproj import CRS, Transformer
import fiona
from fiona.crs import CRS as FionaCRS


# -----------------------------
# Ayarlar (performans için)
# -----------------------------
BATCH_SIZE = 500
DEFAULT_LAYER_NAME = "footprints"
OUTPUT_CRS_EPSG = 4326  # WGS84

FIXED_FIELDS = [
    ("source_file", "str:255"),
    ("building_id", "str:120"),
    ("gml_name", "str:255"),
    ("epsg_source", "int"),

    ("takbisPropertyIdentityNumber", "int"),
    ("totalIndependentSectionCount", "int"),
    ("architecturalProjectConfirmationDate", "str:32"),
    ("elevatorCount", "int"),
    ("roofProjectionArea", "float"),
    ("buildingHeight", "float"),
    ("blockNumber", "str:64"),
    ("parcelNumber", "str:64"),
    ("constructionID", "str:128"),
    ("Atzeminid", "str:128"),

    ("other_gen_attrs", "str"),
]


def _localname(tag: str) -> str:
    if not isinstance(tag, str):
        return ""
    return tag.split("}")[-1] if "}" in tag else tag


_EPSG_RE = re.compile(r"epsg[^0-9]*(\d+)", re.IGNORECASE)


def parse_epsg_from_srsname(srs_name: Optional[str]) -> Optional[int]:
    if not srs_name:
        return None

    m = _EPSG_RE.search(srs_name)
    if m:
        try:
            return int(m.group(1))
        except ValueError:
            return None

    s = srs_name.upper()
    if "CRS84" in s or "CRS:84" in s:
        return 4326

    return None


def safe_int(v: Optional[str]) -> Optional[int]:
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return int(float(v))
        except Exception:
            return None


def safe_float(v: Optional[str]) -> Optional[float]:
    if v is None:
        return None
    v = v.strip()
    if not v:
        return None
    try:
        return float(v)
    except ValueError:
        return None


def ensure_closed_ring(coords: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    if coords and coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def transform_ring(
    coords_xy: Sequence[Tuple[float, float]],
    transformer: Transformer
) -> List[Tuple[float, float]]:
    xs = [c[0] for c in coords_xy]
    ys = [c[1] for c in coords_xy]
    lon, lat = transformer.transform(xs, ys)
    return list(zip(lon, lat))


def extract_gml_id(elem: etree._Element) -> Optional[str]:
    for k, v in elem.attrib.items():
        if _localname(k) == "id":
            return v
    return None


def extract_gml_name(building_elem: etree._Element) -> Optional[str]:
    for e in building_elem.iter():
        if (
            isinstance(e.tag, str)
            and e.tag.startswith("{http://www.opengis.net/gml}")
            and _localname(e.tag) == "name"
        ):
            if e.text and e.text.strip():
                return e.text.strip()
    for e in building_elem.iter():
        if _localname(e.tag) == "name" and e.text and e.text.strip():
            return e.text.strip()
    return None


def extract_gen_attributes(building_elem: etree._Element) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for a in building_elem.iter():
        ln = _localname(a.tag)
        if not ln.endswith("Attribute"):
            continue
        key = a.attrib.get("name")
        if not key:
            continue

        val = None
        for v in a.iter():
            if _localname(v.tag) == "value" and v.text and v.text.strip():
                val = v.text.strip()
                break
        if val is not None:
            out[key] = val
    return out


def find_epsg_in_element(elem: etree._Element) -> Optional[int]:
    for e in elem.iter():
        srs = e.attrib.get("srsName")
        if srs:
            epsg = parse_epsg_from_srsname(srs)
            if epsg:
                return epsg
    return None


def parse_ring_coords(ring_elem: Optional[etree._Element]) -> Optional[List[Tuple[float, float]]]:
    if ring_elem is None:
        return None

    poslist_elem = None
    for e in ring_elem.iter():
        if _localname(e.tag) == "posList" and e.text and e.text.strip():
            poslist_elem = e
            break

    coords: List[Tuple[float, float]] = []

    if poslist_elem is not None:
        nums = [float(x) for x in poslist_elem.text.strip().split()]

        dim = poslist_elem.attrib.get("srsDimension") or poslist_elem.attrib.get("dimension")
        step: Optional[int] = None
        if dim:
            try:
                d = int(dim)
                if d in (2, 3):
                    step = d
            except ValueError:
                step = None

        if step is None:
            if len(nums) % 3 == 0 and len(nums) >= 12:
                step = 3
            elif len(nums) % 2 == 0:
                step = 2
            elif len(nums) % 3 == 0:
                step = 3
            else:
                return None

        for i in range(0, len(nums), step):
            if i + 1 < len(nums):
                coords.append((nums[i], nums[i + 1]))

    else:
        for p in ring_elem.iter():
            if _localname(p.tag) == "pos" and p.text and p.text.strip():
                parts = p.text.strip().split()
                if len(parts) >= 2:
                    coords.append((float(parts[0]), float(parts[1])))

        if not coords:
            for c in ring_elem.iter():
                if _localname(c.tag) == "coordinates" and c.text and c.text.strip():
                    tuples = c.text.strip().split()
                    for t in tuples:
                        parts = t.split(",")
                        if len(parts) >= 2:
                            coords.append((float(parts[0]), float(parts[1])))
                    break

    if not coords or len(coords) < 3:
        return None

    coords = ensure_closed_ring(coords)
    if len(coords) < 4:
        return None
    return coords


def parse_polygon(poly_elem: etree._Element) -> Optional[Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]]:
    exterior_elem = None
    for c in poly_elem:
        if _localname(c.tag) == "exterior":
            exterior_elem = c
            break
    if exterior_elem is None:
        for c in poly_elem.iter():
            if _localname(c.tag) == "exterior":
                exterior_elem = c
                break
    if exterior_elem is None:
        return None

    ring = None
    for r in exterior_elem.iter():
        if _localname(r.tag) == "LinearRing":
            ring = r
            break

    exterior_coords = parse_ring_coords(ring)
    if not exterior_coords:
        return None

    holes: List[List[Tuple[float, float]]] = []
    for interior in poly_elem.iter():
        if _localname(interior.tag) == "interior":
            ring_i = None
            for r in interior.iter():
                if _localname(r.tag) == "LinearRing":
                    ring_i = r
                    break
            hole_coords = parse_ring_coords(ring_i)
            if hole_coords:
                holes.append(hole_coords)

    return exterior_coords, holes


def extract_lod0_footprint_polygons(building_elem: etree._Element) -> List[Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]]:
    polygons: List[Tuple[List[Tuple[float, float]], List[List[Tuple[float, float]]]]] = []
    for fp in building_elem.iter():
        ln = _localname(fp.tag)
        if ln in ("lod0FootPrint", "lod0Footprint"):
            for poly in fp.iter():
                if _localname(poly.tag) == "Polygon":
                    p = parse_polygon(poly)
                    if p:
                        polygons.append(p)
    return polygons


def build_fiona_schema() -> dict:
    return {"geometry": "MultiPolygon", "properties": {name: ftype for name, ftype in FIXED_FIELDS}}


@dataclass
class ExportStats:
    files_total: int
    files_done: int = 0
    features_written: int = 0
    files_failed: int = 0


def export_to_gpkg(
    input_files: Sequence[Path],
    output_gpkg: Path,
    layer_name: str,
    progress_cb,
    stop_flag: threading.Event,
) -> ExportStats:
    start_time = time.time()
    stats = ExportStats(files_total=len(input_files))

    output_gpkg.parent.mkdir(parents=True, exist_ok=True)
    if output_gpkg.exists():
        output_gpkg.unlink()

    schema = build_fiona_schema()
    crs = FionaCRS.from_epsg(OUTPUT_CRS_EPSG)
    transformer_cache: Dict[int, Transformer] = {}

    log_path = output_gpkg.with_suffix(".log.txt")
    if log_path.exists():
        log_path.unlink()

    def log_error(msg: str) -> None:
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(msg.rstrip() + "\n")

    with fiona.open(
        output_gpkg,
        mode="w",
        driver="GPKG",
        layer=layer_name,
        schema=schema,
        crs=crs,
        encoding="UTF-8",
    ) as dst:

        buffer: List[dict] = []

        for idx, gml_path in enumerate(input_files, start=1):
            if stop_flag.is_set():
                break

            current_file = gml_path.name
            file_epsg: Optional[int] = None
            file_failed = False

            try:
                context = etree.iterparse(
                    str(gml_path),
                    events=("start", "end"),
                    recover=True,
                    huge_tree=True,
                )

                for event, elem in context:
                    if event == "start" and file_epsg is None:
                        srs = elem.attrib.get("srsName")
                        if srs:
                            file_epsg = parse_epsg_from_srsname(srs)

                    if event == "end" and _localname(elem.tag) == "Building":
                        building_elem = elem
                        epsg_src = find_epsg_in_element(building_elem) or file_epsg or OUTPUT_CRS_EPSG

                        if epsg_src not in transformer_cache:
                            transformer_cache[epsg_src] = Transformer.from_crs(
                                CRS.from_epsg(epsg_src),
                                CRS.from_epsg(OUTPUT_CRS_EPSG),
                                always_xy=True,
                            )
                        transformer = transformer_cache[epsg_src]

                        polygons = extract_lod0_footprint_polygons(building_elem)
                        if polygons:
                            gen_attrs = extract_gen_attributes(building_elem)
                            building_id = extract_gml_id(building_elem)
                            gml_name = extract_gml_name(building_elem)

                            construction_id = gen_attrs.get("constructionID")
                            atzeminid = construction_id.split("-", 1)[0] if construction_id else None

                            multipoly_coords = []
                            for exterior_xy, holes_xy in polygons:
                                exterior_ll = transform_ring(exterior_xy, transformer)
                                holes_ll = [transform_ring(h, transformer) for h in holes_xy]
                                multipoly_coords.append([exterior_ll] + holes_ll)

                            geom = {"type": "MultiPolygon", "coordinates": multipoly_coords}

                            props = {
                                "source_file": current_file,
                                "building_id": building_id,
                                "gml_name": gml_name,
                                "epsg_source": epsg_src,

                                "takbisPropertyIdentityNumber": safe_int(gen_attrs.get("takbisPropertyIdentityNumber")),
                                "totalIndependentSectionCount": safe_int(gen_attrs.get("totalIndependentSectionCount")),
                                "architecturalProjectConfirmationDate": gen_attrs.get("architecturalProjectConfirmationDate"),
                                "elevatorCount": safe_int(gen_attrs.get("elevatorCount")),
                                "roofProjectionArea": safe_float(gen_attrs.get("roofProjectionArea")),
                                "buildingHeight": safe_float(gen_attrs.get("buildingHeight")),
                                "blockNumber": gen_attrs.get("blockNumber"),
                                "parcelNumber": gen_attrs.get("parcelNumber"),
                                "constructionID": construction_id,
                                "Atzeminid": atzeminid,

                                "other_gen_attrs": json.dumps(gen_attrs, ensure_ascii=False),
                            }

                            buffer.append({"type": "Feature", "geometry": geom, "properties": props})
                            stats.features_written += 1

                            if len(buffer) >= BATCH_SIZE:
                                dst.writerecords(buffer)
                                buffer.clear()

                        building_elem.clear()
                        while building_elem.getprevious() is not None:
                            del building_elem.getparent()[0]

                del context

            except Exception as e:
                file_failed = True
                stats.files_failed += 1
                log_error(f"[FAIL] {gml_path} -> {repr(e)}")

            stats.files_done = idx
            elapsed = time.time() - start_time
            avg = (elapsed / stats.files_done) if stats.files_done else 0.0
            eta = avg * (stats.files_total - stats.files_done)

            progress_cb(stats, elapsed, eta, current_file, file_failed)

        if buffer:
            dst.writerecords(buffer)
            buffer.clear()

    return stats


def fmt_hms(seconds: float) -> str:
    if seconds < 0:
        seconds = 0
    s = int(seconds)
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    return f"{h:02d}:{m:02d}:{sec:02d}"


# -----------------------------
# Streamlit UI
# -----------------------------
st.set_page_config(page_title="CityGML FootPrint → GeoPackage (WGS84)", layout="wide")
st.title("CityGML FootPrint → GeoPackage (WGS84 / EPSG:4326)")

with st.sidebar:
    st.header("Ayarlar")
    layer_name = st.text_input("Layer adı", value=DEFAULT_LAYER_NAME)
    base_name = st.text_input("Çıktı dosya adı (timestamp eklenecek)", value="footprints")
    batch_size = st.number_input("Batch size", min_value=50, max_value=5000, value=BATCH_SIZE, step=50)
    st.caption("Not: Çok büyük batch daha hızlı olabilir ama RAM tüketir.")

# global BATCH_SIZE override
BATCH_SIZE = int(batch_size)

# stop flag
if "stop_flag" not in st.session_state:
    st.session_state.stop_flag = threading.Event()

st.subheader("1) GML Dosyalarını Yükle")
uploads = st.file_uploader("Çoklu .gml seçin", type=["gml"], accept_multiple_files=True)

colA, colB = st.columns(2)
with colA:
    start = st.button("Başlat", disabled=not uploads)
with colB:
    cancel = st.button("Durdur")

log_box = st.empty()
progress_bar = st.progress(0)
status_line = st.empty()

if cancel:
    st.session_state.stop_flag.set()
    status_line.warning("Durdurma istendi (mevcut dosya bitince durur).")

def run_job(temp_dir: Path, input_paths: List[Path], out_path: Path, layer: str):
    logs: List[str] = []

    def progress_cb(stats: ExportStats, elapsed: float, eta: float, current_file: str, file_failed: bool):
        pct = (stats.files_done / stats.files_total * 100) if stats.files_total else 0
        progress_bar.progress(min(stats.files_done / max(stats.files_total, 1), 1.0))
        status_line.info(
            f"%{pct:0.1f} ({stats.files_done}/{stats.files_total}) | "
            f"Geçen: {fmt_hms(elapsed)} | Kalan: {fmt_hms(eta)} | "
            f"Yazılan: {stats.features_written} | Hata: {stats.files_failed} | "
            f"Dosya: {current_file}"
        )
        if file_failed:
            logs.append(f"[HATA] {current_file} okunamadı (log dosyasına yazıldı).")
            log_box.text("\n".join(logs[-30:]))

    stats = export_to_gpkg(
        input_files=input_paths,
        output_gpkg=out_path,
        layer_name=layer,
        progress_cb=progress_cb,
        stop_flag=st.session_state.stop_flag,
    )
    return stats

if start:
    st.session_state.stop_flag.clear()

    # temp workspace
    temp_root = Path(st.session_state.get("temp_root", ""))
    if not temp_root or not temp_root.exists():
        # Streamlit Cloud dahil çalışsın diye yerel bir temp klasör:
        temp_root = Path(".streamlit_tmp")
        temp_root.mkdir(parents=True, exist_ok=True)
        st.session_state.temp_root = str(temp_root)

    # input files -> temp
    job_dir = temp_root / f"job_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    job_dir.mkdir(parents=True, exist_ok=True)

    input_paths: List[Path] = []
    for uf in uploads:
        p = job_dir / uf.name
        p.write_bytes(uf.getbuffer())
        input_paths.append(p)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = job_dir / f"{base_name.strip() or 'footprints'}_{ts}.gpkg"

    st.info(f"{len(input_paths)} dosya yüklendi. Çıktı: {out_path.name}")

    stats = run_job(job_dir, input_paths, out_path, layer_name.strip() or DEFAULT_LAYER_NAME)

    stopped = st.session_state.stop_flag.is_set()
    if stopped:
        st.warning("İşlem durduruldu.")
    else:
        st.success("İşlem tamamlandı.")

    st.write(
        {
            "files_total": stats.files_total,
            "files_done": stats.files_done,
            "features_written": stats.features_written,
            "files_failed": stats.files_failed,
        }
    )

    # download buttons
    gpkg_bytes = out_path.read_bytes() if out_path.exists() else None
    log_path = out_path.with_suffix(".log.txt")
    log_bytes = log_path.read_bytes() if log_path.exists() else None

    if gpkg_bytes:
        st.download_button(
            "GeoPackage indir (.gpkg)",
            data=gpkg_bytes,
            file_name=out_path.name,
            mime="application/geopackage+sqlite3",
        )
    if log_bytes:
        st.download_button(
            "Hata logu indir (.txt)",
            data=log_bytes,
            file_name=log_path.name,
            mime="text/plain",
        )