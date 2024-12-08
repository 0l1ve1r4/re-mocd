use eframe::{run_native, App, CreationContext};
use egui::Context;
use egui_graphs::{DefaultGraphView, Graph, SettingsInteraction, SettingsStyle};

mod  graph;
use graph::{
    high_inter_graph,
    high_intra_graph
};

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
    if true {
        return Graph::from(&high_inter_graph())
    }

    else {
        return Graph::from(&high_intra_graph())
    }

}

fn main() {
    let native_options = eframe::NativeOptions::default();
    run_native(
        "egui_graphs_interactive_demo",
        native_options,
        Box::new(|cc| Ok(Box::new(InteractiveApp::new(cc)))),
    )
    .unwrap();
}