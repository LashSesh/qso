//! Reading and writing graphs as edge lists and GraphML.
//!
//! **Edge lists** are plain text, one edge `u v [weight]` per line with
//! nodes numbered from 0. Lines starting with `#` or `%` are comments,
//! except `# nodes N`, which sets the number of nodes (so that isolated
//! nodes survive a round trip); without it the graph has `max label + 1`
//! nodes. [`to_edge_list`] writes that line.
//!
//! **GraphML** ([graphml.graphdrawing.org](http://graphml.graphdrawing.org/))
//! is read for the common subset: `<node id=…>` and `<edge source=… target=…>`
//! elements, edge weights from a `<data>` element whose key is declared with
//! `attr.name="weight"`, comments and the XML declaration. Node ids may be
//! any strings; nodes are numbered in order of appearance. Directed edges
//! are read as undirected. [`to_graphml`] writes nodes `n0, n1, …` and a
//! weight key.
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
    let _ = writeln!(out, "# {}", g.name().replace('\n', " "));
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
    let mut edges: Vec<(usize, usize, f64)> = Vec::new();
    for (k, raw) in text.lines().enumerate() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(rest) = line.strip_prefix('#').or_else(|| line.strip_prefix('%')) {
            let mut words = rest.split_whitespace();
            if words.next() == Some("nodes") {
                let n: usize = words
                    .next()
                    .and_then(|x| x.parse().ok())
                    .filter(|_| words.next().is_none())
                    .ok_or_else(|| parse_error(format!("line {}: malformed '# nodes N'", k + 1)))?;
                if n > MAX_GENERATED_NODES {
                    return Err(parse_error(format!(
                        "at most {MAX_GENERATED_NODES} nodes are supported"
                    )));
                }
                declared = Some(n);
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
    Graph::from_weighted_edges(n, &edges)
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
    let mut nodes: HashMap<String, usize> = HashMap::new();
    let mut weight_keys: Vec<String> = Vec::new();
    let mut edges: Vec<(String, String, f64)> = Vec::new();
    let mut graph_name: Option<String> = None;
    // Edge being read and the data key whose text is being read.
    let mut open_edge: Option<(String, String, f64)> = None;
    let mut open_data: Option<String> = None;
    let mut seen_graphml = false;
    let mut rest = text;
    while let Some(lt) = rest.find('<') {
        let between = &rest[..lt];
        if let (Some(key), Some(edge)) = (&open_data, open_edge.as_mut())
            && weight_keys.contains(key)
            && !between.trim().is_empty()
        {
            edge.2 = unescape(between.trim())?
                .parse()
                .map_err(|_| parse_error(format!("invalid weight '{}'", between.trim())))?;
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
        let gt = rest
            .find('>')
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
                    weight_keys.push(id.to_string());
                }
            }
            ("graph", false) => {
                if graph_name.is_some() {
                    return Err(parse_error("only one <graph> is supported"));
                }
                graph_name = Some(tag.attr("id").unwrap_or("graph").to_string());
            }
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
                let edge = (source.to_string(), target.to_string(), 1.0);
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
                open_data = Some(tag.attr("key").unwrap_or_default().to_string());
            }
            ("data", true) => open_data = None,
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
    let mut list = Vec::with_capacity(edges.len());
    for (s, t, w) in &edges {
        list.push((index(s)?, index(t)?, *w));
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
