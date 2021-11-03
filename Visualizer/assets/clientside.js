if (!window.dash_clientside) {
     window.dash_clientside = {}
 }

window.dash_clientside.clientside = {

    update_mesh_figure: function (fig_dict, state_data = []) {
        if (!fig_dict) {
            throw window.dash_clientside.PreventUpdate
        }
        let fig_dict_copy = {...fig_dict};

        if ("scene.camera" in state_data) {
            fig_dict_copy["layout"].scene.camera = state_data["scene.camera"];
        }
        return fig_dict_copy;
    },
    update_io_figure: function (fig_dict, state_data) {
        let new_fig_dict = window.dash_clientside.clientside.update_mesh_figure(fig_dict, state_data);
        return new_fig_dict;
    },

    update_3d_figure: function (fig_dict, state_data = []) {
        if (!fig_dict) {
            throw window.dash_clientside.PreventUpdate
        }
        let fig_dict_copy = {...fig_dict};
        return fig_dict_copy;
    },
};