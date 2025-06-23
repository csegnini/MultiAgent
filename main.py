import os
import json
from flask import Flask, request, jsonify, send_from_directory
from dotenv import load_dotenv

# Import the root_agent instance directly from your agent.py file
from agents.agent import root_agent

# Load environment variables (like API keys)
load_dotenv()

app = Flask(__name__)

# This is the directory where your agent saves its output files
OUTPUT_DIR = "data_bases"


@app.route("/")
def index():
    return "The Multi-Agent Data Analysis Pipeline is running.", 200


@app.route("/analyze", methods=["POST"])
def analyze():
    """
    This endpoint receives a query, runs the full agent pipeline,
    and returns the paths to the final results.
    """
    if not request.json or "query" not in request.json:
        return jsonify({"error": "Request must be a JSON with a 'query' field."}), 400

    query = request.json["query"]

    try:
        # --- FINAL, CORRECTED AGENT INVOCATION ---
        # Pass the query directly and provide an empty parent_context
        final_response = root_agent.run_live(query, parent_context={})

        # The ADK LlmAgent wraps the tool output. We need to get it.
        tool_output_str = final_response.get("tool_code_output")
        if not tool_output_str:
            return (
                jsonify(
                    {
                        "error": "Agent did not produce a tool output.",
                        "full_response": final_response,
                    }
                ),
                500,
            )

        # Parse the JSON string from our tool
        results = json.loads(tool_output_str)

        if "error" in results:
            return jsonify(results), 500

        # Create downloadable links for the output files
        csv_filename = os.path.basename(results["correlation_matrix_csv_path"])
        png_filename = os.path.basename(results["correlation_heatmap_png_path"])

        return jsonify(
            {
                "message": results["message"],
                "correlation_matrix_csv": f"/outputs/{csv_filename}",
                "correlation_heatmap_png": f"/outputs/{png_filename}",
            }
        )

    except Exception as e:
        app.logger.error(f"An error occurred during analysis: {e}")
        return jsonify({"error": f"An internal error occurred: {str(e)}"}), 500


@app.route("/outputs/<path:filename>")
def download_file(filename):
    """
    Serves the generated files from the data_bases directory.
    """
    try:
        return send_from_directory(OUTPUT_DIR, filename, as_attachment=True)
    except FileNotFoundError:
        return "File not found.", 404


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
