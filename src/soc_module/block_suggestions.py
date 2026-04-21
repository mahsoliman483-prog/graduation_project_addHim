"""
File: src/soc_module/block_suggestions.py

Convert AI predictions to block rules and send to kernel
"""

import numpy as np
from typing import List, Dict, Tuple

from src.core.kernel_panel import BlockRuleV1, KernelHandle, kp_add_block_rule


class BlockSuggestionEngine:
    """
    Converts AI predictions to BlockRuleV1 objects
    and optionally enforces them via kernel
    """
    
    def __init__(self, auto_block: bool = False):
        """
        Args:
            auto_block: If True, automatically add rules to kernel
        """
        self.auto_block = auto_block
        self.blocked_flows = []
        self.kernel_handle = None
    
    def set_kernel_handle(self, handle: KernelHandle):
        """Set kernel handle for automatic blocking"""
        self.kernel_handle = handle
    
    def make_block_rules(self, 
                        predictions: np.ndarray,
                        meta: List[Dict],
                        threshold: float = 0.5) -> List[BlockRuleV1]:
        """
        Convert predictions to BlockRuleV1 objects.
        
        Args:
            predictions: numpy array of shape (n,) with values 0-1
                        (0=normal, 1=malicious)
            meta: list of metadata dicts from build_feature_batch()
            threshold: confidence threshold (default 0.5)
        
        Returns:
            List of BlockRuleV1 objects for malicious flows
        """
        
        rules = []
        
        for i, pred in enumerate(predictions):
            # Only block if prediction >= threshold
            if pred < threshold:
                continue
            
            flow_meta = meta[i]
            
            # Create block rule
            rule = BlockRuleV1(
                # FlowKey
                ip_version=4 if ':' not in flow_meta['src_ip'] else 6,
                proto=flow_meta['proto'],
                src_ip=flow_meta['src_ip'],
                dst_ip=flow_meta['dst_ip'],
                src_port=flow_meta['src_port'],
                dst_port=flow_meta['dst_port'],
                
                # Rule parameters
                action='BLOCK',
                ttl_ms=60000,  # Block for 60 seconds
                direction_policy='ANY',  # Block both directions
                reason='AI detected malicious traffic',
                reason_code=int(pred * 100)  # Store confidence as reason_code
            )
            
            rules.append(rule)
            
            # Track for logging
            self.blocked_flows.append({
                'flow_key': flow_meta['flow_key'],
                'confidence': float(pred),
                'num_packets': flow_meta['num_packets'],
            })
        
        return rules
    
    def enforce_rules(self, rules: List[BlockRuleV1]) -> Tuple[int, int]:
        """
        Send block rules to kernel for enforcement.
        
        Args:
            rules: List of BlockRuleV1 objects
        
        Returns:
            (successful, failed) - number of rules added successfully
        """
        
        if not self.kernel_handle or not self.kernel_handle.is_running:
            print("⚠️  Warning: Kernel handle not set or not running")
            return 0, len(rules)
        
        successful = 0
        failed = 0
        
        for rule in rules:
            try:
                kp_add_block_rule(self.kernel_handle, rule)
                successful += 1
            except Exception as e:
                print(f"❌ Failed to add block rule: {e}")
                failed += 1
        
        return successful, failed
    
    def get_blocked_flows(self) -> List[Dict]:
        """Get list of recently blocked flows"""
        return self.blocked_flows.copy()
    
    def clear_blocked_flows(self):
        """Clear blocked flows list"""
        self.blocked_flows.clear()


def predictions_to_block_rules(predictions: np.ndarray,
                               meta: List[Dict],
                               threshold: float = 0.5) -> List[BlockRuleV1]:
    """
    Utility function: convert predictions to rules
    (without the engine wrapper)
    
    Args:
        predictions: (n,) array, 0=normal 1=malicious
        meta: metadata list
        threshold: confidence threshold
    
    Returns:
        List of BlockRuleV1
    """
    
    engine = BlockSuggestionEngine()
    return engine.make_block_rules(predictions, meta, threshold)