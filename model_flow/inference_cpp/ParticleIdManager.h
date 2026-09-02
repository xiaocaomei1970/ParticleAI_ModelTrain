#pragma once

/// Session-scoped monotonic particle ID allocator.
/// IDs are never recycled — once assigned, an ID is permanent.
class ParticleIdManager {
public:
    int nextId() { return ++m_counter; }
    int current() const { return m_counter; }
    void reset(int start = 0) { m_counter = start; }

    /// Ensure the counter is at least minId (used when loading existing records)
    void ensureAbove(int minId) { if (m_counter < minId) m_counter = minId; }

private:
    int m_counter = 0;
};
