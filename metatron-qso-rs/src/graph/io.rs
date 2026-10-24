//! Reading and writing graphs as edge lists and GraphML.
//!
//! **Edge lists** are plain text, one edge `u v [weight]` per line with
//! nodes numbered from 0. Lines starting with `#` or `%` are comments,
//! except `# nodes N` (exactly), which sets the number of nodes (so that
//! isolated nodes survive a round trip; without it the graph has
//! `max label + 1` nodes), and `# name: …`. [`to_edge_list`] writes both.
//!
//! **GraphML** ([graphml.graphdrawing.org](http://graphml.graphdrawing.org/))
//! is read for the common subset: `<node id=…>` and `<edge source=… target=…>`
//! elements, edge weights from a `<data>` element whose key is declared with
//! `attr.name="weight"` (and that key's `<default>` for edges without one),
//! comments and the XML declaration. Node ids may be any strings; nodes are
//! numbered in order of appearance. Directed edges are read as undirected;
//! a pair of opposite directed edges with equal weights becomes one edge.
//! Hyperedges, nested graphs, DOCTYPE and CDATA are rejected.
//! [`to_graphml`] writes nodes `n0, n1, …` and a weight key.
//!
//! Both readers reject malformed input with an error and never build more
//! than [`MAX_GENERATED_NODES`] nodes or [`MAX_GENERATED_EDGES`] edges.

use std::collections::HashMap;
use std::fmt::Write as _;

use super::Graph;
use super::generators::{MAX_GENERATED_EDGES, MAX_GENERATED_NODES};
use crate::error::{QsoError, Result};

fn parse_error(msg: impl Into<String>) -> QsoError {
    QsoError::Parse(msg.into())
}

/// Edge list of `g` (see the module documentation).
pub fn to_edge_list(g: &Graph) -> String {
    let mut out = String::new();
    let _ = writeln!(out, "# name: {}", g.name().replace('\n', " "));
    let _ = writeln!(out, "# nodes {}", g.num_nodes());
    for &(u, v, w) in g.edges() {
        if w == 1.0 {
            let _ = writeln!(out, "{u} {v}");
        } else {
            let _ = writeln!(out, "{u} {v} {w:?}");
        }
    }
    out
}

/// Read an edge list (see the module documentation).
pub fn from_edge_list(text: &str) -> Result<Graph> {
    let mut declared: Option<usize> = None;
    let mut name: Option<String> = None;
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for (k, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if line.starts_with('%') {
            continue;
        }
        if let Some(rest) = line.strip_prefix('#') {
            // `# nodes N` (exactly) declares the node count and `# name: …`
            // the name; any other `#` line is a comment.
            let words: Vec<&str> = rest.split_whitespace().collect();
            if let ["nodes", count] = words[..]
                && let Ok(n) = count.parse::<usize>()
            {
                if n > MAX_GENERATED_NODES {
                    return Err(parse_error(format!(
                        "at most {MAX_GENERATED_NODES} nodes are supported"
                    )));
                }
                declared = Some(n);
            } else if let Some(n) = rest.trim_start().strip_prefix("name:") {
                name = Some(n.trim().to_string());
            }
            continue;
        }
        let fields: Vec<&str> = line.split_whitespace().collect();
        let node = |x: &str| -> Result<usize> {
            x.parse::<usize>()
                .ok()
                .filter(|&u| u < MAX_GENERATED_NODES)
                .ok_or_else(|| parse_error(format!("line {}: invalid node '{x}'", k + 1)))
        };
        let (u, v, w) = match fields.as_slice() {
            [u, v] => (node(u)?, node(v)?, 1.0),
            [u, v, w] => (
                node(u)?,
                node(v)?,
                w.parse::<f64>()
                    .map_err(|_| parse_error(format!("line {}: invalid weight '{w}'", k + 1)))?,
            ),
            _ => {
                return Err(parse_error(format!(
                    "line {}: expected 'u v [weight]'",
                    k + 1
                )));
            }
        };
        if edges.len() == MAX_GENERATED_EDGES {
            return Err(parse_error(format!(
                "at most {MAX_GENERATED_EDGES} edges are supported"
            )));
        }
        edges.push((u, v, w));
    }
    let needed = edges
        .iter()
        .map(|&(u, v, _)| u.max(v) + 1)
        .max()
        .unwrap_or(0);
    let n = match declared {
        Some(n) if n < needed => {
            return Err(parse_error(format!(
                "edge to node {} but '# nodes {n}'",
                needed - 1
            )));
        }
        Some(n) => n,
        None => needed,
    };
    let g = Graph::from_weighted_edges(n, &edges)?;
    Ok(match name {
        Some(name) => g.with_name(name),
        None => g,
    })
}

fn escape(text: &str) -> String {
    text.replace('&', "&amp;")
        .replace('<', "&lt;")
        .replace('>', "&gt;")
        .replace('"', "&quot;")
}

/// GraphML document of `g` (see the module documentation).
pub fn to_graphml(g: &Graph) -> String {
    let mut out = String::from(
        "<?xml version=\"1.0\" encoding=\"UTF-8\"?>\n\
         <graphml xmlns=\"http://graphml.graphdrawing.org/xmlns\">\n\
         \x20 <key id=\"weight\" for=\"edge\" attr.name=\"weight\" attr.type=\"double\"/>\n",
    );
    let _ = writeln!(
        out,
        "  <graph id=\"{}\" edgedefault=\"undirected\">",
        escape(g.name())
    );
    for u in 0..g.num_nodes() {
        let _ = writeln!(out, "    <node id=\"n{u}\"/>");
    }
    for &(u, v, w) in g.edges() {
        let _ = writeln!(
            out,
            "    <edge source=\"n{u}\" target=\"n{v}\"><data key=\"weight\">{w:?}</data></edge>"
        );
    }
    out.push_str("  </graph>\n</graphml>\n");
    out
}

/// One XML tag: name (without `/`), attributes, and whether it closes
/// (`</x>`) or is self-closing (`<x/>`).
struct Tag {
    name: String,
    attrs: Vec<(String, String)>,
    closing: bool,
    self_closing: bool,
}

impl Tag {
    fn attr(&self, key: &str) -> Option<&str> {
        self.attrs
            .iter()
            .find(|(k, _)| k == key)
            .map(|(_, v)| v.as_str())
    }
}

fn unescape(text: &str) -> Result<String> {
    let mut out = String::with_capacity(text.len());
    let mut rest = text;
    while let Some(i) = rest.find('&') {
        out.push_str(&rest[..i]);
        let end = rest[i..]
            .find(';')
            .ok_or_else(|| parse_error("unterminated character reference"))?;
        let entity = &rest[i + 1..i + end];
        let ch = match entity {
            "amp" => '&',
            "lt" => '<',
            "gt" => '>',
            "quot" => '"',
            "apos" => '\'',
            _ => entity
                .strip_prefix("#x")
                .map(|h| u32::from_str_radix(h, 16))
                .or_else(|| entity.strip_prefix('#').map(|d| d.parse::<u32>()))
                .and_then(|r| r.ok())
                .and_then(char::from_u32)
                .ok_or_else(|| parse_error(format!("unknown entity '&{entity};'")))?,
        };
        out.push(ch);
        rest = &rest[i + end + 1..];
    }
    out.push_str(rest);
    Ok(out)
}

fn parse_tag(body: &str) -> Result<Tag> {
    let (closing, body) = match body.strip_prefix('/') {
        Some(b) => (true, b),
        None => (false, body),
    };
    let (self_closing, body) = match body.strip_suffix('/') {
        Some(b) => (true, b),
        None => (false, body),
    };
    let body = body.trim();
    let name_end = body.find(char::is_whitespace).unwrap_or(body.len());
    let name = body[..name_end].to_string();
    if name.is_empty() {
        return Err(parse_error("empty tag"));
    }
    let mut attrs = Vec::new();
    let mut rest = body[name_end..].trim_start();
    while !rest.is_empty() {
        let eq = rest
            .find('=')
            .ok_or_else(|| parse_error(format!("malformed attribute in <{name}>")))?;
        let key = rest[..eq].trim().to_string();
        let after = rest[eq + 1..].trim_start();
        let quote = after
            .chars()
            .next()
            .filter(|&c| c == '"' || c == '\'')
            .ok_or_else(|| parse_error(format!("unquoted attribute in <{name}>")))?;
        let close = after[1..]
            .find(quote)
            .ok_or_else(|| parse_error(format!("unterminated attribute in <{name}>")))?;
        attrs.push((key, unescape(&after[1..1 + close])?));
        rest = after[close + 2..].trim_start();
    }
    Ok(Tag {
        name,
        attrs,
        closing,
        self_closing,
    })
}

/// Read a GraphML document (see the module documentation).
pub fn from_graphml(text: &str) -> Result<Graph> {
    /// An edge as read: endpoints, weight if given, and whether directed.
    type RawEdge = (String, String, Option<f64>, bool);
    let mut nodes: HashMap<String, usize> = HashMap::new();
    // Weight keys (by id) with their defaults, in declaration order.
    let mut weight_keys: HashMap<String, Option<f64>> = HashMap::new();
    let mut first_weight_key: Option<String> = None;
    let mut edges: Vec<RawEdge> = Vec::new();
    let mut graph_name: Option<String> = None;
    let mut directed_default = false;
    // Edge being read, the weight key being declared, and the element
    // whose text is being collected (a weight <data> or a <default>).
    let mut open_edge: Option<RawEdge> = None;
    let mut open_key: Option<String> = None;
    let mut text_buf: Option<String> = None;
    let mut seen_graphml = false;
    let parse_weight = |t: &str| -> Result<f64> {
        let t = unescape(t.trim())?;
        t.parse()
            .map_err(|_| parse_error(format!("invalid weight '{t}'")))
    };
    let mut rest = text;
    while let Some(lt) = rest.find('<') {
        if let Some(buf) = text_buf.as_mut() {
            buf.push_str(&rest[..lt]);
        }
        rest = &rest[lt..];
        if let Some(r) = rest.strip_prefix("<!--") {
            let end = r
                .find("-->")
                .ok_or_else(|| parse_error("unterminated comment"))?;
            rest = &r[end + 3..];
            continue;
        }
        if let Some(r) = rest.strip_prefix("<?") {
            let end = r
                .find("?>")
                .ok_or_else(|| parse_error("unterminated processing instruction"))?;
            rest = &r[end + 2..];
            continue;
        }
        if rest.starts_with("<!") {
            return Err(parse_error("DOCTYPE and CDATA are not supported"));
        }
        // The end of the tag: the first '>' outside quoted attribute values.
        let mut quote: Option<char> = None;
        let gt = rest
            .char_indices()
            .find(|&(_, c)| match quote {
                Some(q) => {
                    if c == q {
                        quote = None;
                    }
                    false
                }
                None => {
                    if c == '"' || c == '\'' {
                        quote = Some(c);
                    }
                    c == '>'
                }
            })
            .map(|(i, _)| i)
            .ok_or_else(|| parse_error("unterminated tag"))?;
        let tag = parse_tag(&rest[1..gt])?;
        rest = &rest[gt + 1..];
        match (tag.name.as_str(), tag.closing) {
            ("graphml", false) => seen_graphml = true,
            ("key", false) => {
                if tag.attr("attr.name") == Some("weight")
                    && tag.attr("for").is_none_or(|f| f == "edge" || f == "all")
                {
                    let id = tag
                        .attr("id")
                        .ok_or_else(|| parse_error("<key> without id"))?;
                    weight_keys.insert(id.to_string(), None);
                    first_weight_key.get_or_insert_with(|| id.to_string());
                    if !tag.self_closing {
                        open_key = Some(id.to_string());
                    }
                }
            }
            ("key", true) => open_key = None,
            ("default", false) if open_key.is_some() && !tag.self_closing => {
                text_buf = Some(String::new());
            }
            ("default", true) => {
                if let (Some(key), Some(t)) = (&open_key, text_buf.take()) {
                    weight_keys.insert(key.clone(), Some(parse_weight(&t)?));
                }
            }
            ("graph", false) => {
                if graph_name.is_some() {
                    return Err(parse_error("only one <graph> is supported"));
                }
                graph_name = Some(tag.attr("id").unwrap_or("graph").to_string());
                directed_default = tag.attr("edgedefault") == Some("directed");
            }
            ("hyperedge", _) => return Err(parse_error("hyperedges are not supported")),
            ("node", false) => {
                let id = tag
                    .attr("id")
                    .ok_or_else(|| parse_error("<node> without id"))?;
                if nodes.len() == MAX_GENERATED_NODES {
                    return Err(parse_error(format!(
                        "at most {MAX_GENERATED_NODES} nodes are supported"
                    )));
                }
                let next = nodes.len();
                if nodes.insert(id.to_string(), next).is_some() {
                    return Err(parse_error(format!("node '{id}' declared twice")));
                }
            }
            ("edge", false) => {
                if open_edge.is_some() {
                    return Err(parse_error("nested <edge>"));
                }
                let source = tag
                    .attr("source")
                    .ok_or_else(|| parse_error("<edge> without source"))?;
                let target = tag
                    .attr("target")
                    .ok_or_else(|| parse_error("<edge> without target"))?;
                let directed = match tag.attr("directed") {
                    Some(d) => d == "true",
                    None => directed_default,
                };
                let edge = (source.to_string(), target.to_string(), None, directed);
                if tag.self_closing {
                    edges.push(edge);
                } else {
                    open_edge = Some(edge);
                }
            }
            ("edge", true) => {
                let edge = open_edge
                    .take()
                    .ok_or_else(|| parse_error("</edge> without <edge>"))?;
                edges.push(edge);
            }
            ("data", false) if !tag.self_closing => {
                let key = tag.attr("key").unwrap_or_default();
                if open_edge.is_some() && weight_keys.contains_key(key) {
                    text_buf = Some(String::new());
                }
            }
            ("data", true) => {
                if let (Some(t), Some(edge)) = (text_buf.take(), open_edge.as_mut())
                    && !t.trim().is_empty()
                {
                    edge.2 = Some(parse_weight(&t)?);
                }
            }
            _ => {}
        }
        if edges.len() > MAX_GENERATED_EDGES {
            return Err(parse_error(format!(
                "at most {MAX_GENERATED_EDGES} edges are supported"
            )));
        }
    }
    if !seen_graphml || graph_name.is_none() {
        return Err(parse_error("not a GraphML document with a <graph>"));
    }
    if open_edge.is_some() {
        return Err(parse_error("unterminated <edge>"));
    }
    let index = |id: &str| -> Result<usize> {
        nodes
            .get(id)
            .copied()
            .ok_or_else(|| parse_error(format!("edge to undeclared node '{id}'")))
    };
    let default = first_weight_key
        .and_then(|k| weight_keys[&k])
        .unwrap_or(1.0);
    let mut list = Vec::with_capacity(edges.len());
    // Opposite directed edges with equal weights become one undirected edge.
    let mut directed_seen: HashMap<(usize, usize), f64> = HashMap::new();
    for (s, t, w, directed) in &edges {
        let (u, v, w) = (index(s)?, index(t)?, w.unwrap_or(default));
        if *directed {
            match directed_seen.get(&(v, u)) {
                Some(&back) if back.to_bits() == w.to_bits() => continue,
                Some(_) => {
                    return Err(parse_error(format!(
                        "opposite directed edges '{s}'–'{t}' with different weights"
                    )));
                }
                None => {
                    directed_seen.insert((u, v), w);
                }
            }
        }
        list.push((u, v, w));
    }
    Ok(Graph::from_weighted_edges(nodes.len(), &list)?.with_name(graph_name.unwrap_or_default()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::generators::*;
    use crate::graph::metatron;

    #[test]
    fn round_trips() {
        let mut weighted = Graph::new(5).with_name("w");
        weighted
            .add_weighted_edge(0, 3, -1.5)
            .unwrap()
            .add_weighted_edge(1, 2, 0.1)
            .unwrap();
        for g in [
            petersen(),
            metatron::metatron_cube(),
            Graph::new(3),
            weighted,
        ] {
            let a = from_edge_list(&to_edge_list(&g)).unwrap();
            assert_eq!((a.num_nodes(), a.edges()), (g.num_nodes(), g.edges()));
            let b = from_graphml(&to_graphml(&g)).unwrap();
            assert_eq!((b.num_nodes(), b.edges()), (g.num_nodes(), g.edges()));
            assert_eq!(b.name(), g.name());
        }
    }

    #[test]
    fn review_regressions() {
        // A graph named "nodes …" still round-trips; the name is kept.
        let g = Graph::from_edges(3, &[(0, 1)])
            .unwrap()
            .with_name("nodes 7 and more");
        let back = from_edge_list(&to_edge_list(&g)).unwrap();
        assert_eq!((back.num_nodes(), back.name()), (3, "nodes 7 and more"));
        let doc = "<graphml><graph id=\"nodes\"><node id=\"x\"/></graph></graphml>";
        let h = from_graphml(doc).unwrap();
        assert_eq!(from_edge_list(&to_edge_list(&h)).unwrap().num_nodes(), 1);
        // Ordinary comments that start with "nodes" are comments.
        let e = from_edge_list("% nodes are numbered from 0\n# nodes: see below\n0 1\n").unwrap();
        assert_eq!(e.num_nodes(), 2);
        // <default> of the weight key, comments inside data, '>' in values.
        let doc = r#"<graphml><key id="w" for="edge" attr.name="weight"><default>2.5</default></key>
            <graph edgedefault="undirected"><node id="a>b"/><node id="c"/><node id="d"/>
            <edge source="a>b" target="c"/>
            <edge source="c" target="d"><data key="w">1<!-- x -->5</data></edge></graph></graphml>"#;
        let g = from_graphml(doc).unwrap();
        assert_eq!(g.edges(), &[(0, 1, 2.5), (1, 2, 15.0)]);
        // Opposite directed edges merge when their weights agree.
        let doc = r#"<graphml><graph edgedefault="directed"><node id="a"/><node id="b"/>
            <edge source="a" target="b"/><edge source="b" target="a"/></graph></graphml>"#;
        assert_eq!(from_graphml(doc).unwrap().num_edges(), 1);
        let doc = r#"<graphml><graph><node id="a"/><hyperedge/></graph></graphml>"#;
        assert!(from_graphml(doc).is_err());
    }

    #[test]
    fn reads_foreign_graphml() {
        let text = r#"<?xml version="1.0"?>
<!-- written by another tool -->
<graphml xmlns="http://graphml.graphdrawing.org/xmlns">
  <key id="d0" for="edge" attr.name="weight" attr.type="double"/>
  <key id="d1" for="node" attr.name="label" attr.type="string"/>
  <graph id="G &amp; H" edgedefault="directed">
    <node id="a"><data key="d1">A</data></node>
    <node id='b'/>
    <node id="c"/>
    <edge source="a" target="b"><data key="d0">2.5</data></edge>
    <edge id="e2" source="c" target="b"/>
  </graph>
</graphml>"#;
        let g = from_graphml(text).unwrap();
        assert_eq!(g.name(), "G & H");
        assert_eq!(g.edges(), &[(0, 1, 2.5), (1, 2, 1.0)]);
    }

    #[test]
    fn rejects_malformed_input() {
        for bad in [
            "0 1 2 3",
            "0 x",
            "# nodes 2\n0 5",
            "0 0",
            "0 1\n1 0",
            "0 1 nan",
        ] {
            assert!(from_edge_list(bad).is_err(), "{bad:?}");
        }
        assert!(from_edge_list(&format!("# nodes {}", usize::MAX)).is_err());
        let ok = "<graphml><graph id='g'><node id='a'/><node id='b'/>";
        for bad in [
            "",
            "<graph id='g'/>",
            "<graphml><graph><node id='a'/><edge source='a' target='z'/></graph></graphml>",
            "<graphml><graph><node id='a'/><node id='a'/></graph></graphml>",
            "<graphml><graph><node/></graph></graphml>",
            "<graphml><graph><node id='a",
            "<graphml><!-- unterminated",
            "<graphml><graph><node id=a/></graph></graphml>",
            "<graphml><graph><node id='&bogus;'/></graph></graphml>",
            "<!DOCTYPE x><graphml/>",
        ] {
            assert!(from_graphml(bad).is_err(), "{bad:?}");
        }
        let weight = format!(
            "{ok}<key id='w' attr.name='weight'/><edge source='a' target='b'>\
             <data key='w'>heavy</data></edge></graph></graphml>"
        );
        assert!(from_graphml(&weight).is_err());
    }
}
