use eframe::{run_native, App, CreationContext};
use egui::Context;
use egui_graphs::{DefaultGraphView, Graph, SettingsInteraction, SettingsStyle};

mod graphs;
mod modularity;

const GLOBAL_GRAPH_VIEW: u8 = 0x04; 

pub struct InteractiveApp {
    g: Graph,
}

impl InteractiveApp {
    fn new(_: &CreationContext<'_>) -> Self {
        let g: Graph = generate_graph();
        Self { g }
    }
}

impl App for InteractiveApp {
    fn update(&mut self, ctx: &Context, _: &mut eframe::Frame) {
        egui::CentralPanel::default().show(ctx, |ui| {
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

fn generate_graph() -> Graph {
    match GLOBAL_GRAPH_VIEW {
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