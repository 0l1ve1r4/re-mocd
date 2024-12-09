use eframe::{run_native, App, CreationContext};
use egui::Context;
use egui_graphs::{DefaultGraphView, Graph, SettingsInteraction, SettingsStyle};

mod graphs;
mod modularity;

const GLOBAL_GRAPH_VIEW_DEFAULT: u8 = 0x01; 

pub struct InteractiveApp {
    g: Graph,
    graph_id: u8,
}

impl InteractiveApp {
    fn new(_: &CreationContext<'_>) -> Self {
        let graph_id = GLOBAL_GRAPH_VIEW_DEFAULT;
        let g: Graph = generate_graph(graph_id);
        Self { g, graph_id }
    }

    fn change_graph(&mut self) {
        println!("Graph_id: {}", self.graph_id);

        self.graph_id = match self.graph_id {
            0x01 => 0x02,
            0x02 => 0x03,
            0x03 => 0x04,
            0x04 => 0x01,
            _    => 0x01,
        };
        self.g = generate_graph(self.graph_id);
    }

}

impl App for InteractiveApp {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {

            if ui.button("Change Graph").clicked() {
                self.change_graph();
            }

            let interaction_settings = &SettingsInteraction::new()
                .with_dragging_enabled(true)
                .with_node_clicking_enabled(true)
                .with_node_selection_enabled(true)
                .with_node_selection_multi_enabled(false)
                .with_edge_clicking_enabled(true)
                .with_edge_selection_enabled(true)
                .with_edge_selection_multi_enabled(false)
                .with_edge_selection_enabled(true);
                
            let style_settings = &SettingsStyle::new().with_labels_always(false);
            ui.add(
                &mut DefaultGraphView::new(&mut self.g)
                    .with_styles(style_settings)
                    .with_interactions(interaction_settings),
            );
        });
    }

}

fn generate_graph(graph_view: u8) -> Graph {
    match graph_view {
        0x01 => return Graph::from(&graphs::perfectly_communities().0),
        0x02 => return Graph::from(&graphs::create_complete_mixing().0),
        0x03 => return Graph::from(&graphs::create_single_community().0),
        0x04 => return Graph::from(&graphs::create_sparse_communities().0),
        _    => return Graph::from(&graphs::create_empty_graph().0),
    }

}

fn main() {
    let native_options = eframe::NativeOptions::default();
    run_native(
        "MOCD - Graph Viewer",
        native_options,
        Box::new(|cc| Ok(Box::new(InteractiveApp::new(cc)))),
    )
    .unwrap();
}