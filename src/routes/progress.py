from flask import Blueprint, jsonify, request
from flask_cors import cross_origin
import json
import os
from datetime import datetime

progress_bp = Blueprint('progress', __name__)

# In-memory storage for progress tracking (in production, use Redis or database)
execution_progress = {}

@progress_bp.route('/api/progress/<execution_id>', methods=['GET'])
@cross_origin()
def get_progress(execution_id):
    """Get progress status for a specific execution"""
    try:
        if execution_id not in execution_progress:
            return jsonify({
                'error': 'Execution not found',
                'execution_id': execution_id
            }), 404
        
        progress_data = execution_progress[execution_id]
        return jsonify(progress_data)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@progress_bp.route('/api/progress/<execution_id>/update', methods=['POST'])
@cross_origin()
def update_progress(execution_id):
    """Update progress status for a specific execution"""
    try:
        data = request.get_json()
        
        if execution_id not in execution_progress:
            execution_progress[execution_id] = {
                'execution_id': execution_id,
                'status': 'initializing',
                'progress': 0,
                'current_step': '',
                'logs': [],
                'agents': {},
                'start_time': datetime.now().isoformat(),
                'estimated_completion': None,
                'error': None
            }
        
        # Update progress data
        progress_data = execution_progress[execution_id]
        
        if 'status' in data:
            progress_data['status'] = data['status']
        
        if 'progress' in data:
            progress_data['progress'] = min(100, max(0, data['progress']))
        
        if 'current_step' in data:
            progress_data['current_step'] = data['current_step']
        
        if 'log' in data:
            log_entry = {
                'timestamp': datetime.now().isoformat(),
                'message': data['log'],
                'level': data.get('level', 'info'),
                'agent': data.get('agent', 'manager')
            }
            progress_data['logs'].append(log_entry)
            
            # Keep only last 100 log entries
            if len(progress_data['logs']) > 100:
                progress_data['logs'] = progress_data['logs'][-100:]
        
        if 'agent_status' in data:
            agent_name = data['agent_status']['name']
            progress_data['agents'][agent_name] = {
                'status': data['agent_status']['status'],
                'progress': data['agent_status'].get('progress', 0),
                'current_task': data['agent_status'].get('current_task', ''),
                'last_update': datetime.now().isoformat()
            }
        
        if 'estimated_completion' in data:
            progress_data['estimated_completion'] = data['estimated_completion']
        
        if 'error' in data:
            progress_data['error'] = data['error']
            progress_data['status'] = 'error'
        
        if 'result' in data:
            progress_data['result'] = data['result']
            progress_data['status'] = 'completed'
            progress_data['progress'] = 100
            progress_data['end_time'] = datetime.now().isoformat()
        
        return jsonify({'success': True, 'progress': progress_data})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@progress_bp.route('/api/progress/<execution_id>/logs', methods=['GET'])
@cross_origin()
def get_logs(execution_id):
    """Get detailed logs for a specific execution"""
    try:
        if execution_id not in execution_progress:
            return jsonify({
                'error': 'Execution not found',
                'execution_id': execution_id
            }), 404
        
        progress_data = execution_progress[execution_id]
        return jsonify({
            'execution_id': execution_id,
            'logs': progress_data.get('logs', []),
            'total_logs': len(progress_data.get('logs', []))
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@progress_bp.route('/api/progress', methods=['GET'])
@cross_origin()
def list_executions():
    """List all active executions"""
    try:
        executions = []
        for execution_id, data in execution_progress.items():
            executions.append({
                'execution_id': execution_id,
                'status': data.get('status', 'unknown'),
                'progress': data.get('progress', 0),
                'current_step': data.get('current_step', ''),
                'start_time': data.get('start_time'),
                'agents_count': len(data.get('agents', {}))
            })
        
        return jsonify({
            'executions': executions,
            'total': len(executions)
        })
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

