#include <igl/readMESH.h>
#include <igl/opengl/glfw/Viewer.h>
#include <igl/unproject_onto_mesh.h>
#include <igl/opengl/glfw/imgui/ImGuiPlugin.h>
#include <igl/opengl/glfw/imgui/ImGuiMenu.h>
#include <igl/opengl/glfw/imgui/ImGuiHelpers.h>
#include <igl/opengl/glfw/imgui/ImGuizmoWidget.h>


#include <igl/barycentric_coordinates.h>

#include "SimulatorCore/Simulator.h"

#include <iostream>


Simulator *simulator;
MeshData _mesh;

// Face IDs
std::vector<int> constraintID, displacedID, selectedID, vertIDs;

enum Colors
{
    BLUE,
    RED,
    GREEN
};

void load_mesh(std::string filePath, igl::opengl::glfw::Viewer& viewer, MeshData& _mesh, bool viewMesh)
{

    igl::readMESH(filePath, _mesh.V, _mesh.T, _mesh.F);

    simulator->SetUpMeshes(_mesh);

    viewer.data().clear();
    viewer.data().set_mesh(_mesh.V, _mesh.F);
    viewer.core().align_camera_center(_mesh.V);
    viewer.data().add_label(viewer.data().V.row(0) + viewer.data().V_normals.row(0).normalized() * 0.005, "Hello World!");
    viewer.data().invert_normals = true;
    // Initialize Face Selection w/ New Data
    _mesh.C = Eigen::MatrixXd::Constant(_mesh.F.rows(), 3, 1);
    viewer.data().set_colors(_mesh.C);
}

void draw_viewer_menu(igl::opengl::glfw::Viewer &viewer, MeshData& _mesh, std::string title, bool viewMesh)
{

    float w = ImGui::GetContentRegionAvail().x;
    float p = ImGui::GetStyle().FramePadding.x;
    if (ImGui::Button(title.c_str(), ImVec2((w - p) / 2.f, 0)))
    {
        std::string fname = igl::file_dialog_open();

        if (fname.length() == 0)
            return;

        load_mesh(fname, viewer, _mesh, viewMesh);
    }
    
}

void set_color(igl::opengl::glfw::Viewer& viewer, int fid, Colors color)
{
    switch (color)
    {
    case RED:
        _mesh.C.row(fid) << 1, 0, 0; break;
    case BLUE:
        _mesh.C.row(fid) << 0, 0, 1; break;
    case GREEN:
        _mesh.C.row(fid) << 0, 1, 0; break;
    }
    viewer.data().set_colors(_mesh.C);
}

bool face_selection(igl::opengl::glfw::Viewer& viewer, int s, int c)
{
    int fid;
    Eigen::Vector3f bc;
    // Cast a ray in the view direction starting from the mouse position
    double x = viewer.current_mouse_x;
    double y = viewer.core().viewport(3) - viewer.current_mouse_y;
    if (igl::unproject_onto_mesh(Eigen::Vector2f(x, y), viewer.core().view,
        viewer.core().proj, viewer.core().viewport, _mesh.V, _mesh.F, fid, bc))
    {
        // paint hit red
        _mesh.C.row(fid) << 1, 0, 0;
        selectedID.push_back(fid);
        viewer.data().set_colors(_mesh.C);
        return true;
    }
    return false;
}

void displace_verts(std::vector<int> FIDs, double x, double y, double z)
{
    for (auto ID : FIDs)
    {
        // Get vertex IDs
        int v1 = _mesh.F(ID, 0);
        int v2 = _mesh.F(ID, 1);
        int v3 = _mesh.F(ID, 2);

        _mesh.V(v1, 0) += x; _mesh.V(v1, 1) += y; _mesh.V(v1, 2) += z;
        _mesh.V(v2, 0) += x; _mesh.V(v2, 1) += y; _mesh.V(v2, 2) += z;
        _mesh.V(v3, 0) += x; _mesh.V(v3, 1) += y; _mesh.V(v3, 2) += z;
    }
}

void extract_constraint_verts()
{
    for (auto ID : constraintID)
    {
        int v1 = _mesh.F(ID, 0);
        int v2 = _mesh.F(ID, 1);
        int v3 = _mesh.F(ID, 2);
        vertIDs.push_back(v1); vertIDs.push_back(v2); vertIDs.push_back(v3);
    }
    for (auto ID : displacedID)
    {
        int v1 = _mesh.F(ID, 0);
        int v2 = _mesh.F(ID, 1);
        int v3 = _mesh.F(ID, 2);
        vertIDs.push_back(v1); vertIDs.push_back(v2); vertIDs.push_back(v3);
    }
}


int main(int argc, char* argv[])
{
    // Energy Selection
    const char* energies[] = { "ARAP", "Symmetric Drichilet" };
    static const char* current_energy = "ARAP";

    simulator = new Simulator();

    double x = 0, y = 0, z = 0;

    // Init the viewer
    igl::opengl::glfw::Viewer viewer;

    // Attach a menu plugin
    igl::opengl::glfw::imgui::ImGuiPlugin plugin;
    viewer.plugins.push_back(&plugin);
    igl::opengl::glfw::imgui::ImGuiMenu menu;
    plugin.widgets.push_back(&menu);


    // Customize the menu
    double doubleVariable = 0.1f; // Shared between two menus

    // Add content to the default menu window
    menu.callback_draw_viewer_menu = [&]()
    {
        // Draw parent menu content
        if (ImGui::CollapsingHeader("Mesh Selection", ImGuiTreeNodeFlags_DefaultOpen))
        {
            draw_viewer_menu(viewer, _mesh, "Input Mesh ", true);
        }

        if (ImGui::BeginCombo("##Energy To Minimize", current_energy))
        {
            for (int n = 0; n < IM_ARRAYSIZE(energies); n++)
            {
                bool is_selected = (current_energy == energies[n]);
                if (ImGui::Selectable(energies[n], is_selected))
                {
                    current_energy = energies[n];
                    simulator->ChangeEnergy(n);
                }
                if (is_selected)
                    ImGui::SetItemDefaultFocus();
            }
            ImGui::EndCombo();
        }

        if (ImGui::CollapsingHeader("Distort Mesh", ImGuiTreeNodeFlags_DefaultOpen))
        {
            ImGui::InputDouble("x:", &x); ImGui::InputDouble("y:", &y); ImGui::InputDouble("z:", &z);

            if (ImGui::Button("Set Constrained"))
            {
                constraintID = selectedID;
                for (auto ID : constraintID) set_color(viewer, ID, GREEN);
                viewer.data().set_colors(_mesh.C);
                selectedID.clear();
            }
            if (ImGui::Button("Displace Selected"))
            {
                displacedID = selectedID;
                for (auto ID : displacedID) set_color(viewer, ID, BLUE);
                displace_verts(displacedID, x, y, z);
                viewer.data().clear();
                viewer.data().set_mesh(_mesh.V, _mesh.F);
                viewer.core().align_camera_center(_mesh.V);
                viewer.data().add_label(viewer.data().V.row(0) + viewer.data().V_normals.row(0).normalized() * 0.005, "Hello World!");
                viewer.data().invert_normals = true;
                viewer.data().set_colors(_mesh.C);
                selectedID.clear();
            }
        }

        if (ImGui::Button("Initiate Solver"))
        {
            extract_constraint_verts();
            simulator->constraintVertIDs = vertIDs;
            simulator->InputDeformedMesh(_mesh);
        }

    };

    viewer.callback_mouse_down = face_selection;

    viewer.launch();
}
